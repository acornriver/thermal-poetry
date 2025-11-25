import sys
import time
import random
import math
import pygame
import numpy as np
from collections import deque
import threading
from PIL import Image, ImageOps
import sounddevice as sd


# Try imports for hardware/NLP, fallback if missing for development
try:
    from konlpy.tag import Okt
    HAS_KONLPY = True
except Exception as e:
    HAS_KONLPY = False
    print(f"Warning: konlpy import failed ({e}). Using mock NLP.")

try:
    from escpos.printer import Usb
    HAS_ESCPOS = True
except ImportError:
    HAS_ESCPOS = False
    print("Warning: python-escpos not found. Using mock printer.")


# --- Configuration ---
SCREEN_WIDTH = 2560  # 2K QHD resolution
SCREEN_HEIGHT = 1440
BG_COLOR = (0, 0, 0)  # Black
FONT_COLOR = (255, 255, 255)
FONT_NAME = "malgun"  # Windows default Korean font
FONT_SIZE = 45  # Scaled for 2K
IDLE_TIMEOUT = 10.0  # Seconds to switch to Floating state
EXPLOSION_DURATION = 3.0  # Seconds for Deconstruction state

# --- States ---
STATE_FLOATING = "FLOATING"
STATE_TYPING = "TYPING"
STATE_DECONSTRUCTION = "DECONSTRUCTION"
STATE_PRINTING = "PRINTING"  # New state for print button held

STATE_PRINTING = "PRINTING"  # New state for breath trigger

# --- Animation Config ---
RECOIL_DURATION = 0.3  # Seconds for anticipation
RECOIL_FORCE = 5.0     # Speed to move right during recoil
LAUNCH_ACCEL = 1.5     # Acceleration for launch (pixels/frame^2)
MIC_THRESHOLD = 0.1    # Volume threshold for breath detection


class Word:
    """Represents a floating word in the visualizer."""
    def __init__(self, text, x, y, font):
        self.text = text
        self.x = x
        self.y = y
        self.font = font
        
        # Physics
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        
        # Ocean-like wave motion
        self.phase_offset = random.uniform(0, math.pi * 2)
        self.wave_speed = random.uniform(0.5, 0.8)  # Slower for deeper feel
        self.wave_amplitude_y = random.uniform(15, 25)  # Deeper vertical motion
        self.wave_amplitude_x = random.uniform(5, 10)  # Subtle horizontal drift
        
        # Rotation
        self.rotation = random.uniform(-5, 5)  # Initial tilt
        self.rotation_speed = random.uniform(-0.1, 0.1)  # Slow rotation
        
        # Base position (for wave calculation)
        self.base_x = x
        self.base_y = y

        # --- CACHING: Pre-render surfaces ---
        # 1. Shadow
        shadow_color = (40, 40, 40)
        self.original_shadow = self.font.render(self.text, True, shadow_color)
        
        # 2. Main Text
        self.original_text = self.font.render(self.text, True, FONT_COLOR)
    
    def update(self, current_time):
        """Update word position with pure physics-based motion."""
        # Simple physics: position changes by velocity
        self.x += self.vx
        self.y += self.vy
        
        # Gentle natural damping (air resistance)
        self.vx *= 0.995
        self.vy *= 0.995
        
        # Update rotation
        self.rotation += self.rotation_speed
    
    def draw(self, screen):
        # Minimalist rendering: clean typography only
        
        # 1. Draw subtle shadow for depth
        # Rotate pre-rendered shadow
        shadow_surf = pygame.transform.rotate(self.original_shadow, self.rotation)
        shadow_rect = shadow_surf.get_rect(center=(int(self.x + 2), int(self.y + 2)))
        screen.blit(shadow_surf, shadow_rect)
        
        # 2. Draw main text with rotation
        # Rotate pre-rendered text
        text_surf = pygame.transform.rotate(self.original_text, self.rotation)
        text_rect = text_surf.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(text_surf, text_rect)

class Particle:
    """Small geometric particle for explosion effects."""
    def __init__(self, x, y):
        self.x = x + random.uniform(-20, 20) # Slight offset from center
        self.y = y + random.uniform(-20, 20)
        self.vx = random.uniform(-0.2, 0.2) # Very slow drift
        self.vy = random.uniform(-0.2, 0.2)
        self.life = 1.0
        self.decay = random.uniform(0.01, 0.03) # Slow fade
        self.size = random.randint(6, 15) # Larger size for visibility
        self.shape = random.choice(['circle', 'rect', 'triangle'])
        self.color_val = random.randint(200, 255) # Bright
        self.color = (self.color_val, self.color_val, self.color_val)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        # No size shrinking, just fade out

    def draw(self, screen):
        if self.life <= 0:
            return
        
        alpha = int(self.life * 255)
        s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        
        c = (*self.color, alpha)
        
        if self.shape == 'circle':
            pygame.draw.circle(s, c, (self.size, self.size), self.size, width=1)
        elif self.shape == 'rect':
            pygame.draw.rect(s, c, (0, 0, self.size*2, self.size*2), width=1)
        elif self.shape == 'triangle':
            points = [(self.size, 0), (0, self.size*2), (self.size*2, self.size*2)]
            pygame.draw.polygon(s, c, points, width=1)
            
        screen.blit(s, (int(self.x - self.size), int(self.y - self.size)))

class NLPManager:
    """Handles text analysis."""
    def __init__(self):
        self.okt = None
        if HAS_KONLPY:
            try:
                self.okt = Okt()
            except Exception as e:
                print(f"Warning: Okt initialization failed (Java missing?): {e}")
                self.okt = None

    def extract_keywords(self, text):
        """Extract words or deconstruct text artistically."""
        if not text.strip():
            return []
            
        # 1. Try Konlpy (if available)
        if self.okt:
            try:
                # pos() returns list of (word, tag)
                results = self.okt.pos(text, stem=True)
                keywords = [word for word, tag in results if tag in ['Noun', 'Verb', 'Adjective']]
                return keywords
            except Exception as e:
                print(f"NLP Error: {e}")
                # Fallthrough to fallback
        
        # 2. Fallback: Artistic Deconstruction
        # If text has spaces, assume user meant to space it
        if " " in text:
            return text.split()
        
        # 3. If no spaces (long block), artistically fragment it
        # "나는오늘죽어서피를남겼다" -> Random chunks of 2-4 chars
        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= 3:
                chunks.append(remaining)
                break
                
            # Random length 2 to 4
            cut_len = random.randint(2, 4)
            chunk = remaining[:cut_len]
            chunks.append(chunk)
            remaining = remaining[cut_len:]
            
        return chunks

    def switch_state(self, new_state):
        self.state = new_state
        print(f"State switched to: {new_state}")
        
        if new_state == STATE_DECONSTRUCTION:
            self.deconstruction_start_time = time.time()
            # Process input
            full_text = self.input_text + self.composition_text
            extracted = self.nlp.extract_keywords(full_text)
            
            # Add to floating words
            for w in extracted:
                self.add_floating_word(w)
            
            # Sound: Read the words (TTS)
            self.speak_text(extracted)
                
            # Print - REMOVED
            # self.printer.print_receipt(extracted)
            
            # Clear input
            self.input_text = ""
            self.composition_text = ""

    def speak_text(self, words):
        """Use Mac 'say' command to read text."""
        if not words:
            return
        
        text_to_speak = " ".join(words)
        try:
            # Run in background
            subprocess.Popen(["say", text_to_speak])
        except Exception as e:
            print(f"TTS Error: {e}")

class PrinterManager:
    """Handles thermal printer output asynchronously."""
    def __init__(self):
        self.printer = None
        self.print_queue = deque()
        self.is_printing = False
        self.lock = threading.Lock()
        
        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._print_worker, daemon=True)
        self.worker_thread.start()

        if HAS_ESCPOS:
            try:
                # Attempt to connect to Epson TM-T88V via USB (vendor 0x04b8, product 0x0e15)
                from escpos.printer import Usb
                self.printer = Usb(0x04b8, 0x0e15)
                print("[PRINTER] Connected to TM-T88V (USB)")
            except Exception as e:
                # If USB connection fails (e.g., missing libusb), fall back to a dummy printer
                print(f"[PRINTER] USB connection failed ({e}), using mock printer.")
                try:
                    from escpos.printer import Dummy
                    self.printer = Dummy()
                    print("[PRINTER] Using mock printer (Dummy)")
                except Exception as e2:
                    print(f"[PRINTER] Failed to initialize dummy printer: {e2}")
        else:
            print("[PRINTER] escpos library not available, printer disabled.")
    
    def _print_worker(self):
        """Background thread to handle print jobs."""
        while True:
            if self.print_queue:
                job = None
                with self.lock:
                    if self.print_queue:
                        job = self.print_queue.popleft()
                
                if job:
                    self.is_printing = True
                    try:
                        self._execute_print(job)
                    except Exception as e:
                        print(f"[PRINTER] Error in print worker: {e}")
                    finally:
                        self.is_printing = False
            
            time.sleep(0.1)  # Prevent busy waiting

    def _execute_print(self, words_data):
        """Actual printing logic (runs in background thread)."""
        if not self.printer:
            return

        try:
            # Initialize printer settings for high speed
            # self.printer._raw(b'\x1b\x40') # Initialize
            
            # Header
            self.printer.text("Thermal Poetry\n")
            self.printer.text("-" * 30 + "\n")
            
            for item in words_data:
                raw = item.text if isinstance(item, Word) else str(item)
                
                # Echo effect logic
                max_offset = 4
                for offset in range(max_offset + 1):
                    line = " " * offset + raw
                    self.printer.text(f"{line}\n")
            
            # Feed and Full Cut
            self.printer.text("\n\n\n")
            self.printer.cut(mode='FULL') 
            
            print(f"[PRINTER] Printed with echo effect: {[item.text if isinstance(item, Word) else item for item in words_data]}")
            
        except Exception as e:
            print(f"[PRINTER] Print failed ({e}); falling back to console output.")
            for item in words_data:
                raw = item.text if isinstance(item, Word) else str(item)
                print(raw)

    def print_receipt(self, words_data):
        """Add print job to queue (Non-blocking)."""
        if not self.printer:
            print("[PRINTER] No printer available – skipping.")
            return

        with self.lock:
            self.print_queue.append(words_data)
        print(f"[PRINTER] Job queued. Queue size: {len(self.print_queue)}")


class MicrophoneManager:
    """Handles microphone input for breath detection with dynamic threshold."""
    def __init__(self):
        self.stream = None
        self.current_volume = 0
        self.noise_floor = 0.2  # Initial estimate of background noise
        self.alpha = 0.01       # Slow adaptation rate for noise floor
        
        # FFT Energy tracking
        self.low_energy = 0.0
        self.high_energy = 0.0
        self.voice_energy = 0.0
        
        try:
            # Initialize input stream
            self.stream = sd.InputStream(callback=self.audio_callback,
                                       channels=1,
                                       samplerate=44100,
                                       blocksize=1024)
            self.stream.start()
            print("[MIC] Microphone initialized successfully")
        except Exception as e:
            print(f"[MIC] Failed to initialize: {e}")
            self.stream = None

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(status)
        # Calculate RMS volume
        volume = np.linalg.norm(indata) * 10
        self.current_volume = volume
        
        # Dynamic Noise Floor Adaptation
        # We assume background noise is relatively steady and lower than "blowing"
        # Only adapt if the current volume is not a sudden spike (like blowing)
        # We use a multiplier (e.g., 2.0 or 3.0) to define what "quiet" means relative to current floor
        if volume < self.noise_floor * 2.5:
            self.noise_floor = self.noise_floor * (1 - self.alpha) + volume * self.alpha
            
        # Frequency Analysis (FFT)
        # Perform FFT to distinguish breath (low freq noise) from typing (high freq clicks)
        try:
            # Apply Hanning window to reduce spectral leakage
            windowed_data = indata[:, 0] * np.hanning(len(indata))
            fft_data = np.fft.rfft(windowed_data)
            fft_freq = np.fft.rfftfreq(len(windowed_data), d=1/44100)
            
            # Calculate energy in bands
            # Breath: Strong < 500Hz
            # Typing/Click: Broadband or High > 1000Hz
            
            mag = np.abs(fft_data)
            
            # Low Freq Energy (0 - 300Hz) - Rumble/Noise
            low_mask = (fft_freq < 300)
            low_energy = np.sum(mag[low_mask])
            
            # Voice Band Energy (300Hz - 3400Hz) - Human Speech
            voice_mask = (fft_freq >= 300) & (fft_freq < 3400)
            voice_energy = np.sum(mag[voice_mask])
            
            self.low_energy = low_energy
            self.voice_energy = voice_energy
            
        except Exception as e:
            print(f"FFT Error: {e}")
            self.low_energy = 0
            self.voice_energy = 0

    def is_speaking(self):
        """Check if input matches human speech characteristics."""
        if not self.stream:
            return False
            
        # Criteria for Speech:
        # 1. Volume > Threshold
        # 2. Significant energy in Voice Band (300-3400Hz)
        
        # Threshold: Adjusted for voice projection
        threshold = max(self.noise_floor + 0.5, 0.6)
        is_loud_enough = self.current_volume > threshold
        
        # Voice Presence Check
        # Voice should have significant energy compared to low-end rumble
        # Avoid division by zero
        total_energy = self.low_energy + self.voice_energy + 0.001
        voice_ratio = self.voice_energy / total_energy
        
        # Speech usually has balanced or voice-band dominant energy
        # Wind/Blow is very low-end dominant (ratio < 0.2)
        # Speech should be > 0.3 approx
        is_voice_freq = voice_ratio > 0.3
        
        return is_loud_enough and is_voice_freq

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class ThermalPoetryApp:
    def create_printer_sound(self, duration=0.04):
        """Generate a mechanical printer/dot-matrix sound."""
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        
        buf = []
        for i in range(n_samples):
            t = i / sample_rate
            # Sharp envelope
            envelope = math.exp(-t * 50)
            
            # Noise-based sound (like mechanical impact)
            noise = random.uniform(-1, 1)
            
            # Add some high-frequency buzz (like motor)
            buzz = math.sin(2 * math.pi * 3000 * t) * 0.3
            
            # Combine
            value = (noise * 0.7 + buzz * 0.3) * envelope
            
            final_value = int(32767 * 0.2 * value)
            final_value = max(-32767, min(32767, final_value))
            buf.append([final_value, final_value])
        
        sound = pygame.sndarray.make_sound(np.array(buf, dtype=np.int16))
        return sound

    def create_typing_sound(self, base_freq=800, duration=0.08, timbre='bell'):
        """Generate a typing sound with varied timbre."""
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        
        buf = []
        for i in range(n_samples):
            t = i / sample_rate
            # Envelope: Sharp attack, exponential decay
            envelope = math.exp(-t * 15)
            
            value = 0
            
            if timbre == 'bell':
                # Bell-like sound with harmonics
                value += math.sin(2 * math.pi * base_freq * t)
                value += 0.5 * math.sin(2 * math.pi * base_freq * 2.5 * t)
                value += 0.3 * math.sin(2 * math.pi * base_freq * 4.2 * t)
            elif timbre == 'pluck':
                # Plucked string
                value += math.sin(2 * math.pi * base_freq * t)
                value += 0.3 * math.sin(2 * math.pi * base_freq * 2 * t)
                # Add slight noise
                value += random.uniform(-0.1, 0.1)
            elif timbre == 'click':
                # Percussive click with noise
                value = random.uniform(-1, 1) * (1 - t / duration)
                value += 0.5 * math.sin(2 * math.pi * base_freq * t)
            elif timbre == 'soft':
                # Soft, mellow tone
                value += math.sin(2 * math.pi * base_freq * t)
                value += 0.4 * math.sin(2 * math.pi * base_freq * 1.5 * t)
            
            # Apply envelope and convert to int16
            final_value = int(32767 * 0.25 * envelope * value)
            final_value = max(-32767, min(32767, final_value))
            buf.append([final_value, final_value])
        
        sound = pygame.sndarray.make_sound(np.array(buf, dtype=np.int16))
        return sound

    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Setup Screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Thermal Poetry")
        
        # Setup Font
        self.font = self.get_korean_font()

        # Managers
        self.nlp = NLPManager()
        self.printer = PrinterManager()
        self.mic = MicrophoneManager()
        
        # Application State
        self.state = STATE_FLOATING
        self.last_input_time = time.time()
        
        # Text Input State
        self.input_text = "" 
        self.composition_text = ""
        pygame.key.start_text_input() # Enable IME
        
        self.floating_words = deque(maxlen=50) # FIFO structure with max limit
        
        # Timer for Deconstruction state
        self.deconstruction_start_time = 0
        
        # Sound FX - Create a pool of varied sounds for word appearance
        self.typing_sounds = []
        
        # Musical scale (pentatonic for pleasant harmony)
        scale_notes = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]  # C, D, E, G, A, C
        timbres = ['bell', 'pluck', 'click', 'soft']
        
        # Generate 12 different sounds for word appearance
        for i in range(12):
            freq = random.choice(scale_notes)
            timbre = random.choice(timbres)
            sound = self.create_typing_sound(freq, duration=random.uniform(0.06, 0.1), timbre=timbre)
            self.typing_sounds.append(sound)
        
        # Printer sound (mechanical, for printing action)
        self.printer_sound = self.create_printer_sound()
        
        # Word appearance tracking for sound sync
        self.word_appear_queue = []  # [(word, appear_time), ...]
        self.last_word_sound_time = 0
        
        # Animation State
        self.printing_start_time = 0
        self.wind_pressure = 0.0 # 0.0 to 1.0
        self.smoothed_force = 0.0 # Smoothed force strength to prevent jerking
        self.typing_cooldown = 0 # Timestamp when typing last happened
        self.printed_words = []  # Words that have been printed
        # UI warning timer for left-ward drift (0..1)
        self.warning_timer = 0.0
        
        # Particle System
        self.particles = []
        
        # Load some initial words
        initial_words = ["당신의", "언어는", "어항", "속에서", "타고", "있나요?"]
        for w in initial_words:
            self.add_floating_word(w)

    def get_korean_font(self):
        """Attempt to find a system font that supports Korean."""
        # List of font names to try - prioritizing serif fonts used by writers/artists
        font_candidates = [
            "nanummyeongjo", "nanummyeongjocoding",  # Nanum Myeongjo - elegant serif
            "notoserifcjkkr", "notoserifkr",  # Noto Serif - professional serif
            "applemyungjo", "applegothic",  # Mac serif fonts
            "batang", "batangche",  # Windows traditional serif
            "applesdgothicneo",  # Fallback to gothic if no serif available
            "malgun", "malgungothic",  # Windows
            "nanumgothic",  # Fallback gothic
        ]
        
        available_fonts = pygame.font.get_fonts()
        
        selected_font = None
        for name in font_candidates:
            if name in available_fonts:
                selected_font = name
                break
        
        if selected_font:
            print(f"Selected font: {selected_font}")
            return pygame.font.SysFont(selected_font, FONT_SIZE)
        else:
            print("Warning: No Korean font found. Text may appear broken.")
            # Fallback to default system font
            return pygame.font.SysFont(None, FONT_SIZE)
    
    def add_floating_word(self, text):
        x = random.randint(100, SCREEN_WIDTH - 100)
        y = random.randint(100, SCREEN_HEIGHT - 100)
        word = Word(text, x, y, self.font)
        self.floating_words.append(word)
        return word

    def log_to_file(self, text):
        """Log user input to log.md with timestamp."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n## {timestamp}\n{text}\n"
        
        log_path = "/Users/acornriver/Documents/GitHub/thermal-poetry/log.md"
        
        try:
            # Append to log file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            print(f"[LOG] Saved to log.md: {text[:20]}...")
        except Exception as e:
            print(f"[LOG] Failed to save: {e}")

    def switch_state(self, new_state):
        self.state = new_state
        print(f"State switched to: {new_state}")
        
        if new_state == STATE_DECONSTRUCTION:
            self.deconstruction_start_time = time.time()
            # Process input
            full_text = self.input_text + self.composition_text
            extracted = self.nlp.extract_keywords(full_text)
            
            # Log the original text to file
            if full_text.strip():
                self.log_to_file(full_text)
            
            # Queue words for sound-synced appearance
            current_time = time.time()
            new_words = []  # Store Word objects for printing
            for i, w in enumerate(extracted):
                # Stagger appearance times
                appear_time = current_time + (i * 0.15)  # 150ms between each word
                self.word_appear_queue.append((w, appear_time))
            
            # Note: We'll collect the actual Word objects after they're created
            # For now, print will happen when button is released with collected words
            
            # Clear input
            self.input_text = ""
            self.composition_text = ""

    def play_word_sounds(self):
        """Play typing sounds synchronized with word appearance."""
        if not self.word_appear_queue:
            return
            
        current_time = time.time()
        
        # Check if any words should appear now
        words_to_add = []
        remaining_queue = []
        
        for word, appear_time in self.word_appear_queue:
            if current_time >= appear_time:
                words_to_add.append(word)
                # Play random sound from pool
                random.choice(self.typing_sounds).play()
            else:
                remaining_queue.append((word, appear_time))
        
        # Add words to floating
        for word_text in words_to_add:
            new_word = self.add_floating_word(word_text)
            self.create_explosion(new_word.x, new_word.y)
        
        # Update queue
        self.word_appear_queue = remaining_queue

    def create_explosion(self, x, y):
        """Create a minimal burst of particles at x, y."""
        for _ in range(3): # Only 3 particles per word for minimalism
            self.particles.append(Particle(x, y))

    def handle_input(self, event):
        self.last_input_time = time.time()
        
        # Mouse input removed (Microphone replaces button)
        
        # Check for typing to set cooldown AND switch state
        if event.type in [pygame.KEYDOWN, pygame.TEXTINPUT, pygame.TEXTEDITING]:
            self.typing_cooldown = time.time() + 0.5 # 0.5s cooldown
            
            # Force switch to TYPING state if not already
            # This ensures typing takes priority over blowing/printing animation
            if self.state != STATE_TYPING and self.state != STATE_DECONSTRUCTION:
                self.switch_state(STATE_TYPING)
        
        if self.state == STATE_FLOATING:

            # Any key switches to Typing
            if event.type in [pygame.KEYDOWN, pygame.TEXTINPUT, pygame.TEXTEDITING]:
                self.switch_state(STATE_TYPING)
                
        if self.state == STATE_TYPING:
            if event.type == pygame.TEXTINPUT:
                self.input_text += event.text
                self.composition_text = "" # Clear composition
                
            elif event.type == pygame.TEXTEDITING:
                # Only show composition if there's actual text
                # This prevents showing incomplete jamo (ㅊ, ㅓ, ㅅ)
                if event.text and len(event.text.strip()) > 0:
                    self.composition_text = event.text
                else:
                    self.composition_text = ""
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.switch_state(STATE_DECONSTRUCTION)
                elif event.key == pygame.K_BACKSPACE:
                    # Only handle backspace if no composition is active (IME handles composition backspace)
                    if not self.composition_text:
                        self.input_text = self.input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    pass

    def update(self):
        current_time = time.time()
        
        # State Transitions
        if self.state == STATE_TYPING:
            # Check idle timeout
            if current_time - self.last_input_time > IDLE_TIMEOUT:
                self.switch_state(STATE_FLOATING)
                
        elif self.state == STATE_DECONSTRUCTION:
            # Play word appearance sounds
            self.play_word_sounds()
            
            # Check explosion duration
            if current_time - self.deconstruction_start_time > EXPLOSION_DURATION:
                self.switch_state(STATE_FLOATING)

        # Update Particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        # Update Logic based on State
        if self.state == STATE_FLOATING:
            # Check Microphone Trigger (Voice)
            if self.mic.is_speaking():
                print(f"[MIC] Voice detected! Volume: {self.mic.current_volume:.2f}")
                self.switch_state(STATE_PRINTING)
                self.printing_start_time = time.time()
                
            for word in self.floating_words:
                word.update(current_time)
                
        elif self.state == STATE_DECONSTRUCTION:
            # Just let them float naturally, no jitter
            for word in self.floating_words:
                word.update(current_time)
                
        elif self.state == STATE_PRINTING:
            words_to_remove = []
            
            # Continuous Voice Physics
            
            # 1. Check Priority (Typing > Mic)
            is_typing = current_time < self.typing_cooldown
            is_speaking = False
            
            if not is_typing:
                is_speaking = self.mic.is_speaking()
            
            # 2. Update "Voice Pressure"
            # Pressure builds up when speaking, decays when not
            if is_speaking:
                # Faster buildup for speech (it's intermittent)
                self.wind_pressure += 0.08
            else:
                # Slower decay to bridge gaps between words
                self.wind_pressure -= 0.015
                
            # Clamp pressure
            self.wind_pressure = max(0.0, min(1.0, self.wind_pressure))
            
            if self.wind_pressure > 0:
                self.last_input_time = current_time # Keep state alive
            
            # 3. Apply Physics based on Pressure
            # Pressure 0.0 - 0.3: Recoil (Move Right)
            # Pressure 0.3 - 1.0: Launch (Move Left)
            
            words_moving = False
            
            for word in self.floating_words:
                
                # Calculate Force based on Pressure
                if self.wind_pressure > 0:
                    # Direct Launch (No Recoil)
                    # Force is negative (Left)
                    # Scale force by pressure directly
                    # Reduced speed (2.5 -> 1.5) for gentler flow
                    launch_strength = self.wind_pressure * 0.5 
                    word.vx -= launch_strength * random.uniform(0.8, 1.2)
                    
                    # Vertical turbulence
                    word.vy += random.uniform(-0.5, 0.5) * launch_strength

                # Apply Friction (Air Resistance)
                word.vx *= 0.92 
                word.vy *= 0.90
                
                # Update Position
                word.x += word.vx
                word.y += word.vy
                
                # Check if still moving significantly
                if abs(word.vx) > 0.1:
                    words_moving = True
                
                # Check bounds (Off-screen Left)
                if word.x < -200:
                    words_to_remove.append(word)

            # 4. Remove collected words and Print
            if words_to_remove:
                self.printer.print_receipt(words_to_remove)
                self.printer_sound.play()
                
                # Add to printed_words for preview
                self.printed_words.extend(words_to_remove)

                for word in words_to_remove:
                    if word in self.floating_words:
                        self.floating_words.remove(word)
            
            # 5. Return to Floating if idle
            # If pressure is zero AND words have stopped moving
            if self.wind_pressure <= 0 and not words_moving:
                if current_time - self.last_input_time > 1.0:
                    self.switch_state(STATE_FLOATING)
            
            # 5. Return to Floating if idle
            # If pressure is zero AND words have stopped moving
            if self.wind_pressure <= 0 and not words_moving:
                if current_time - self.last_input_time > 1.0:
                    self.switch_state(STATE_FLOATING)
            
            # After applying forces, detect leftward movement for UI cue
            self.left_moving = any(word.vx < -0.02 for word in self.floating_words)
            # Update warning timer: fade in when drifting left, fade out otherwise
            if self.left_moving:
                self.warning_timer = min(self.warning_timer + 0.07, 1.0)
            else:
                # Faster fade-out when not moving left
                self.warning_timer *= 0.80
                # Ensure full reset when timer is very low
                if self.warning_timer < 0.02:
                    self.warning_timer = 0.0
            # --- Organic Feedback: Gentle Acceleration ---
            # Apply subtle force with smooth fade-out to prevent jerking
            
            # Calculate target force strength
            if is_speaking or self.wind_pressure > 0:
                target_force = (self.mic.current_volume * 0.2) + (self.wind_pressure * 0.3)
            else:
                target_force = 0.0
            
            # Smoothly interpolate to target (prevents sudden jerks)
            smoothing = 0.15  # Lower = smoother but slower response
            self.smoothed_force += (target_force - self.smoothed_force) * smoothing
            
            # Only apply force if it's significant
            if self.smoothed_force > 0.01:
                for word in self.floating_words:
                    # Apply oscillating force (not position change)
                    force_x = math.sin(current_time * 3.0 + word.phase_offset) * self.smoothed_force
                    force_y = math.cos(current_time * 2.5 + word.phase_offset) * self.smoothed_force
                    
                    # Add to velocity (acceleration), not position
                    word.vx += force_x * 0.05
                    word.vy += force_y * 0.05
            
            # Always apply damping for smooth deceleration
            for word in self.floating_words:
                word.vx *= 0.98
                word.vy *= 0.98


    def draw(self):
        # Motion blur effect: semi-transparent black overlay instead of full clear
        # This creates ethereal trails as words move
        trail_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        trail_surface.fill((0, 0, 0, 15))  # Very subtle fade (15/255 opacity)
        self.screen.blit(trail_surface, (0, 0))
        
        # Draw Particles (Global)
        for p in self.particles:
            p.draw(self.screen)
        
        if self.state == STATE_FLOATING or self.state == STATE_PRINTING:
            
            # Screen Shake (Global Offset)
            shake_x = 0
            shake_y = 0
            if self.state == STATE_PRINTING and self.wind_pressure > 0.5:
                shake_intensity = (self.wind_pressure - 0.5) * 5.0
                shake_x = random.uniform(-shake_intensity, shake_intensity)
                shake_y = random.uniform(-shake_intensity, shake_intensity)

            for word in self.floating_words:
                # Draw with shake offset
                original_x, original_y = word.x, word.y
                word.x += shake_x
                word.y += shake_y
                word.draw(self.screen)
                word.x, word.y = original_x, original_y # Restore
            
            # Visual feedback for Voice (Subtle Circle)
            # Only show if actually speaking (above threshold)
            is_speaking = self.mic.is_speaking()
            
            if is_speaking:
                # Draw indicator
                indicator_radius = int(self.mic.current_volume * 100)
                s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                
                # Very subtle grey
                color = (150, 150, 150, 50)
                
                # Draw circle with width=1 for very thin outline
                pygame.draw.circle(s, color, (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), indicator_radius, width=1)
                self.screen.blit(s, (0,0))
            
            # Render leftward movement warning if needed
            # Artistic warning UI when words drift left
            if getattr(self, 'left_moving', False) or self.warning_timer > 0.01:
                # Fade & slide animation based on warning_timer (0..1)
                progress = self.warning_timer  # 0 = invisible, 1 = fully visible
                # Text
                warning_text = "언어가 감열소각장으로 이동중입니다..."
                warning_surf = self.font.render(warning_text, True, (255, 120, 120))
                # Apply alpha for fade effect
                warning_surf.set_alpha(int(200 * progress))
                # Background glow (soft semi-transparent rectangle)
                glow_surf = pygame.Surface((warning_surf.get_width() + 20,
                                          warning_surf.get_height() + 12),
                                         pygame.SRCALPHA)
                glow_color = (30, 0, 0, int(120 * progress))
                pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=6)
                # Combine glow + text
                combined = pygame.Surface(glow_surf.get_size(), pygame.SRCALPHA)
                combined.blit(glow_surf, (0, 0))
                combined.blit(warning_surf, (10, 6))
                # Slide from right: start slightly off-screen and move left as it appears
                slide_offset = int((1 - progress) * 40)  # 40px slide distance
                warning_rect = combined.get_rect()
                warning_rect.centerx = SCREEN_WIDTH - warning_rect.width // 2 - 20 - slide_offset
                warning_rect.centery = SCREEN_HEIGHT // 2
                self.screen.blit(combined, warning_rect)
            
            # Guidance Text (Dynamic & Fading)
            # Only show in Floating state
            if self.state == STATE_FLOATING:
                alpha = int(abs(math.sin(time.time() * 1.5)) * 255)
                
                # Alternate between two messages every 5 seconds
                cycle = int(time.time() / 5) % 2
                
                if len(self.floating_words) < 3:
                    if cycle == 0:
                        guide_text = "당신의 언어는 어디서 살고 있나요?"
                    else:
                        guide_text = "아무 텍스트나 적어 보세요..."
                else:
                    if cycle == 0:
                        guide_text = "떠다니는 말들을 천천히 읽어보세요..."
                    else:
                        guide_text = "아무 텍스트나 적어 보세요..."
                    
                guide_surf = self.font.render(guide_text, True, (100, 100, 100))
                guide_surf.set_alpha(alpha)
                guide_rect = guide_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                self.screen.blit(guide_surf, guide_rect)
                
        elif self.state == STATE_TYPING:
            # Show Typing Text
            display_text = self.input_text + self.composition_text
            if not display_text:
                display_text = "..."
            
            # Render Text
            text_surf = self.font.render(display_text, True, FONT_COLOR)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)
            
            # Guidance
            guide_surf = self.font.render("Enter를 눌러 해체하기", True, (80, 80, 80))
            guide_rect = guide_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            self.screen.blit(guide_surf, guide_rect)
                
        elif self.state == STATE_DECONSTRUCTION:
            # Show words scattering
            for word in self.floating_words:
                word.draw(self.screen)
                
        pygame.display.flip()

    def run(self):
        """Main application loop (windowed mode)."""
        # Set to windowed mode for development, can be changed to FULLSCREEN
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Global key to exit (F12 or something, since F11 is fullscreen)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F12:
                    running = False
                    
                self.handle_input(event)
            
            self.update()
            self.draw()
            clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = ThermalPoetryApp()
    app.run()
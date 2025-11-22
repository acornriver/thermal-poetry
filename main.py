import sys
import time
import random
import math
import pygame
import numpy as np
from collections import deque

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
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BG_COLOR = (0, 0, 0)  # Black
FONT_COLOR = (255, 255, 255)
FONT_NAME = "malgun"  # Windows default Korean font
FONT_SIZE = 30
IDLE_TIMEOUT = 10.0  # Seconds to switch to Floating state
EXPLOSION_DURATION = 3.0  # Seconds for Deconstruction state

# --- States ---
STATE_FLOATING = "FLOATING"
STATE_TYPING = "TYPING"
STATE_DECONSTRUCTION = "DECONSTRUCTION"
STATE_PRINTING = "PRINTING"  # New state for print button held

# --- Print Button Config ---
PRINT_BUTTON_WIDTH = 200
PRINT_BUTTON_HEIGHT = 60
PRINT_BUTTON_Y = SCREEN_HEIGHT - 100
PRINT_BUTTON_X = (SCREEN_WIDTH - PRINT_BUTTON_WIDTH) // 2
PRINT_PULL_SPEED = 8.0  # Speed at which words are pulled left

class Word:
    """Represents a floating word in the visualizer."""
    def __init__(self, text, x, y, font):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.font = font
        
        # Random velocity (Slower for "heavier" feel)
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        
        # Rotation
        self.angle = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-0.5, 0.5)
        
        # "Alive" properties (Breathing/Squirming)
        self.breath_phase_x = random.uniform(0, 6.28)
        self.breath_phase_y = random.uniform(0, 6.28)
        self.breath_speed = random.uniform(0.5, 1.5)
        
        # Render initial surface
        self.original_surf = self.font.render(self.text, True, FONT_COLOR)
        self.base_rect = self.original_surf.get_rect()
        self.rect = self.base_rect.copy()
        self.rect.center = (self.x, self.y)

    def update(self):
        """Update position, rotation, and breathing."""
        t = time.time()
        
        # --- 1. Organic "Water" Physics (Slower, Heavier) ---
        # Vertical Bobbing: Slower frequency, larger amplitude
        self.y += math.sin(t * 0.8 + self.x * 0.005) * 0.2
        
        # Horizontal Drift: Very slow
        self.x += self.vx * 0.3
        
        # Gentle Rotation: Oscillate slowly
        target_angle = math.sin(t * 0.3 + self.x * 0.005) * 10
        self.angle = target_angle
        
        # Screen Wrapping
        if self.x > SCREEN_WIDTH + 100: self.x = -100
        elif self.x < -100: self.x = SCREEN_WIDTH + 100
        if self.y > SCREEN_HEIGHT + 100: self.y = -100
        elif self.y < -100: self.y = SCREEN_HEIGHT + 100
            
        self.rect.center = (self.x, self.y)

    def draw(self, surface):
        """Draw the rotated word."""
        # Rotate surface
        rotated_surf = pygame.transform.rotate(self.original_surf, self.angle)
        new_rect = rotated_surf.get_rect(center=self.rect.center)
        
        surface.blit(rotated_surf, new_rect)

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
                
            # Print
            self.printer.print_receipt(extracted)
            
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
    """Handles thermal printer output."""
    def __init__(self):
        self.printer = None
        if HAS_ESCPOS:
            try:
                # Replace with actual Vendor/Product ID for Epson TM-T88V
                # Example: 0x04b8, 0x0202 (Need to verify specific model IDs)
                # For now, we'll wrap in try-except to avoid crashing if not connected
                # self.printer = Usb(0x04b8, 0x0202, 0, 0x81, 0x03)
                pass 
            except Exception as e:
                print(f"Printer connection failed: {e}")

    def print_receipt(self, words):
        """Print the list of words."""
        print(f"[PRINTER] Printing: {words}") # Console log for debugging
        
        if self.printer:
            try:
                self.printer.text("Thermal Poetry\n")
                self.printer.text("-" * 30 + "\n")
                for word in words:
                    self.printer.text(f"{word}\n")
                self.printer.cut()
            except Exception as e:
                print(f"Print failed: {e}")

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
        
        # Print Button
        self.print_button_rect = pygame.Rect(PRINT_BUTTON_X, PRINT_BUTTON_Y, 
                                              PRINT_BUTTON_WIDTH, PRINT_BUTTON_HEIGHT)
        self.print_button_pressed = False
        self.printed_words = []  # Words that have been printed
        
        # Load some initial words
        self.last_input_time = time.time()
        
        # Text Input State
        self.input_text = "" 
        self.composition_text = ""
        pygame.key.start_text_input() # Enable IME
        
        self.floating_words = deque(maxlen=50) # FIFO structure with max limit
        
        # Timer for Deconstruction state
        self.deconstruction_start_time = 0
        
        # Sound FX - Create a pool of varied sounds
        self.typing_sounds = []
        
        # Musical scale (pentatonic for pleasant harmony)
        scale_notes = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]  # C, D, E, G, A, C
        timbres = ['bell', 'pluck', 'click', 'soft']
        
        # Generate 12 different sounds
        for i in range(12):
            freq = random.choice(scale_notes)
            timbre = random.choice(timbres)
            sound = self.create_typing_sound(freq, duration=random.uniform(0.06, 0.1), timbre=timbre)
            self.typing_sounds.append(sound)
        
        # Word appearance tracking for sound sync
        self.word_appear_queue = []  # [(word, appear_time), ...]
        self.last_word_sound_time = 0
        
        # Print Button
        self.print_button_rect = pygame.Rect(PRINT_BUTTON_X, PRINT_BUTTON_Y, 
                                              PRINT_BUTTON_WIDTH, PRINT_BUTTON_HEIGHT)
        self.print_button_pressed = False
        self.printed_words = []  # Words that have been printed
        
        # Load some initial words
        initial_words = ["기억", "시간", "흐름", "침묵", "소리", "바람"]
        for w in initial_words:
            self.add_floating_word(w)

    def get_korean_font(self):
        """Attempt to find a system font that supports Korean."""
        # List of font names to try (Mac, Windows, Linux)
        font_candidates = [
            "applesdgothicneo", "applegothic", # Mac
            "malgun", "malgungothic", # Windows
            "dotum", "gulim", # Windows Legacy
            "nanumgothic", "nanummyeongjo", # Linux/Common
            "notosanskj", # Common
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

    def switch_state(self, new_state):
        self.state = new_state
        print(f"State switched to: {new_state}")
        
        if new_state == STATE_DECONSTRUCTION:
            self.deconstruction_start_time = time.time()
            # Process input
            full_text = self.input_text + self.composition_text
            extracted = self.nlp.extract_keywords(full_text)
            
            # Queue words for sound-synced appearance
            current_time = time.time()
            for i, w in enumerate(extracted):
                # Stagger appearance times
                appear_time = current_time + (i * 0.15)  # 150ms between each word
                self.word_appear_queue.append((w, appear_time))
                
            # Print
            self.printer.print_receipt(extracted)
            
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
        for word in words_to_add:
            self.add_floating_word(word)
        
        # Update queue
        self.word_appear_queue = remaining_queue

    def handle_input(self, event):
        self.last_input_time = time.time()
        
        # Handle print button (mouse)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                if self.print_button_rect.collidepoint(mouse_pos):
                    self.print_button_pressed = True
                    if self.state == STATE_FLOATING:
                        self.state = STATE_PRINTING
                        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self.print_button_pressed:
                    self.print_button_pressed = False
                    if self.state == STATE_PRINTING:
                        # Print collected words
                        if self.printed_words:
                            self.printer.print_receipt(self.printed_words)
                            self.printed_words = []
                        self.state = STATE_FLOATING
        
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

        # Update Logic based on State
        if self.state == STATE_FLOATING:
            for word in self.floating_words:
                word.update()
                
        elif self.state == STATE_DECONSTRUCTION:
            # Just let them float naturally, no jitter
            for word in self.floating_words:
                word.update()
                
        elif self.state == STATE_PRINTING:
            # Pull words to the left
            words_to_remove = []
            for word in self.floating_words:
                # Pull left
                word.x -= PRINT_PULL_SPEED
                
                # Check if word has reached the left edge
                if word.x < -100:
                    words_to_remove.append(word)
                    self.printed_words.append(word.text)
                    # Play printer sound when word is "collected"
                    self.printer_sound.play()
                
                # Still update vertical motion
                word.update()
            
            # Remove collected words
            for word in words_to_remove:
                self.floating_words.remove(word)

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        if self.state == STATE_FLOATING:
            for word in self.floating_words:
                word.draw(self.screen)
            
            # Guidance Text (Blinking)
            if int(time.time()) % 2 == 0:
                guide_text = "아무 키나 눌러 당신의 시를 시작하세요"
                guide_surf = self.font.render(guide_text, True, (100, 100, 100))
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
        
        elif self.state == STATE_PRINTING:
            # Show words flowing to the left
            for word in self.floating_words:
                word.draw(self.screen)
        
        # Draw Print Button (minimal design)
        if self.state in [STATE_FLOATING, STATE_PRINTING]:
            # Simple button background
            button_color = (50, 50, 50) if not self.print_button_pressed else (70, 70, 70)
            pygame.draw.rect(self.screen, button_color, self.print_button_rect, border_radius=5)
            
            # Thin border
            border_color = (100, 100, 100)
            pygame.draw.rect(self.screen, border_color, self.print_button_rect, width=1, border_radius=5)
            
            # Simple text
            button_text = "인쇄 ←"
            button_surf = self.font.render(button_text, True, (200, 200, 200))
            button_rect = button_surf.get_rect(center=self.print_button_rect.center)
            self.screen.blit(button_surf, button_rect)
            
            # Simple counter (no animation)
            if self.state == STATE_PRINTING and self.printed_words:
                count_text = f"{len(self.printed_words)}"
                count_surf = pygame.font.SysFont(None, 20).render(count_text, True, (120, 120, 120))
                count_rect = count_surf.get_rect(center=(SCREEN_WIDTH // 2, PRINT_BUTTON_Y - 30))
                self.screen.blit(count_surf, count_rect)
                
        pygame.display.flip()

    def run(self):
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

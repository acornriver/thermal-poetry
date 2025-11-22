import sys
import time
import random
import math
import pygame
import numpy as np
from collections import deque
from PIL import Image

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
        self.x = x
        self.y = y
        self.font = font
        
        # Physics
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        
        # Water-like motion
        self.time_offset = random.uniform(0, math.pi * 2)
        self.bob_speed = random.uniform(0.5, 1.5)
        self.bob_amplitude = random.uniform(10, 30)
        
        # Rotation
        self.rotation = 0
        self.rotation_speed = random.uniform(-0.3, 0.3)
        
        # Generate procedural image based on word
        self.image = self.generate_word_image()
        self.image_offset_x = random.randint(-50, 50)
        self.image_offset_y = random.randint(-50, 50)
        self.image_alpha = random.randint(100, 200)
    
    def generate_word_image(self):
        """Generate a procedural image based on word characteristics."""
        # Use word properties to determine image characteristics
        word_hash = sum(ord(c) for c in self.text)
        
        # Image size
        size = 60 + (word_hash % 40)
        
        # Create surface
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Choose pattern based on word length
        pattern_type = len(self.text) % 4
        
        # Grayscale color (varying brightness based on word)
        brightness = 100 + (word_hash % 100)
        
        center = size // 2
        
        if pattern_type == 0:
            # Concentric circles
            num_circles = 3 + (word_hash % 3)
            for i in range(num_circles):
                radius = size // 2 - (i * size // (num_circles * 2))
                alpha = 150 - (i * 40)
                color = (brightness, brightness, brightness, alpha)
                pygame.draw.circle(surf, color, (center, center), radius, 2)
        
        elif pattern_type == 1:
            # Radiating lines
            num_lines = 5 + (word_hash % 5)
            for i in range(num_lines):
                angle = (360 / num_lines) * i + (word_hash % 360)
                end_x = center + int(size // 2 * math.cos(math.radians(angle)))
                end_y = center + int(size // 2 * math.sin(math.radians(angle)))
                color = (brightness, brightness, brightness, 120)
                pygame.draw.line(surf, color, (center, center), (end_x, end_y), 2)
        
        elif pattern_type == 2:
            # Scattered dots
            num_dots = 8 + (word_hash % 8)
            for i in range(num_dots):
                dot_x = (word_hash * (i + 1) * 17) % size
                dot_y = (word_hash * (i + 1) * 23) % size
                dot_size = 2 + (i % 4)
                color = (brightness, brightness, brightness, 150)
                pygame.draw.circle(surf, color, (dot_x, dot_y), dot_size)
        
        else:
            # Geometric shape
            num_sides = 3 + (word_hash % 4)
            points = []
            for i in range(num_sides):
                angle = (360 / num_sides) * i
                px = center + int(size // 3 * math.cos(math.radians(angle)))
                py = center + int(size // 3 * math.sin(math.radians(angle)))
                points.append((px, py))
            color_fill = (brightness, brightness, brightness, 100)
            color_outline = (brightness, brightness, brightness, 180)
            pygame.draw.polygon(surf, color_fill, points)
            pygame.draw.polygon(surf, color_outline, points, 2)
        
        return surf
    
    def update(self):
        # Gentle bobbing motion
        t = time.time() + self.time_offset
        self.y += math.sin(t * self.bob_speed) * 0.5
        self.x += math.cos(t * self.bob_speed * 0.7) * 0.3
        
        # Slow rotation
        self.rotation += self.rotation_speed
        
        # Screen wrapping
        if self.x > SCREEN_WIDTH + 50:
            self.x = -50
        elif self.x < -50:
            self.x = SCREEN_WIDTH + 50
        if self.y > SCREEN_HEIGHT + 50:
            self.y = -50
        elif self.y < -50:
            self.y = SCREEN_HEIGHT + 50
    
    def draw(self, screen):
        # Draw procedural image first (behind text)
        if self.image:
            img_x = int(self.x + self.image_offset_x)
            img_y = int(self.y + self.image_offset_y)
            
            # Rotate image slightly
            rotated_img = pygame.transform.rotate(self.image, self.rotation * 0.5)
            img_rect = rotated_img.get_rect(center=(img_x, img_y))
            
            # Apply alpha
            rotated_img.set_alpha(self.image_alpha)
            screen.blit(rotated_img, img_rect)
        
        # Draw text
        text_surf = self.font.render(self.text, True, FONT_COLOR)
        text_rect = text_surf.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(text_surf, text_rect)

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
                # Mock printer for development
                from escpos.printer import Dummy
                self.printer = Dummy()
                print("[PRINTER] Using mock printer (Dummy)")
            except Exception as e:
                print(f"[PRINTER] Failed to initialize: {e}")
    
    def print_receipt(self, words_data):
        """Print words with their images.
        
        Args:
            words_data: List of Word objects or list of strings
        """
        if not self.printer:
            print("[PRINTER] No printer available")
            return
            
        try:
            self.printer.text("Thermal Poetry\n")
            self.printer.text("-" * 30 + "\n")
            
            for item in words_data:
                # Check if it's a Word object or just a string
                if isinstance(item, Word):
                    word_text = item.text
                    word_image = item.image
                    
                    # Print image if available
                    if word_image:
                        # Convert pygame surface to PIL Image
                        img_str = pygame.image.tostring(word_image, 'RGB')
                        pil_img = Image.frombytes('RGB', word_image.get_size(), img_str)
                        
                        # Convert to grayscale
                        pil_img = pil_img.convert('L')
                        
                        # Resize to fit printer width (max 384 pixels for most thermal printers)
                        max_width = 200
                        if pil_img.width > max_width:
                            ratio = max_width / pil_img.width
                            new_size = (max_width, int(pil_img.height * ratio))
                            pil_img = pil_img.resize(new_size)
                        
                        # Print image
                        try:
                            self.printer.image(pil_img)
                        except Exception as e:
                            print(f"[PRINTER] Image print failed: {e}")
                    
                    # Print text
                    self.printer.text(f"{word_text}\n")
                else:
                    # Just a string
                    self.printer.text(f"{item}\n")
            
            self.printer.cut()
            print(f"[PRINTER] Printing: {[item.text if isinstance(item, Word) else item for item in words_data]}")
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
                        # Print collected words with their images
                        if self.printed_words:
                            # Find the actual Word objects from floating_words
                            words_to_print = []
                            for word_text in self.printed_words:
                                # Find matching Word object
                                for word_obj in self.floating_words:
                                    if word_obj.text == word_text:
                                        words_to_print.append(word_obj)
                                        break
                            
                            # If we couldn't find Word objects, just print text
                            if not words_to_print:
                                self.printer.print_receipt(self.printed_words)
                            else:
                                self.printer.print_receipt(words_to_print)
                            
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

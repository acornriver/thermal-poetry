import pygame
import sys

# Initialize pygame
pygame.init()

# Screen dimensions (same as main app)
SCREEN_WIDTH = 3840  # 4K resolution (or adjust as needed)
SCREEN_HEIGHT = 2160

# Create screen (off-screen surface for rendering)
screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

# Font settings – use a Korean-supporting font if available
def get_font(size=45):
    # Try common Korean fonts, fallback to default
    candidates = ["nanummyeongjo", "notoserifcjkkr", "applemyungjo", "batang", "malgun", None]
    for name in candidates:
        try:
            if name:
                return pygame.font.SysFont(name, size)
            else:
                return pygame.font.SysFont(None, size)
        except Exception:
            continue
    return pygame.font.SysFont(None, size)

font = get_font()

# Sample words that would be printed on the receipt
sample_words = ["당신의", "언어는", "어항", "속에서", "타고", "있나요?"]

# Echo effect parameters
MAX_OFFSET = 4  # number of echo lines

# Render receipt preview

def render_receipt_preview(surface, words, font):
    # White background
    surface.fill((255, 255, 255))
    y = 20
    line_height = font.get_height() + 4
    # Header
    header = font.render("Thermal Poetry", True, (0, 0, 0))
    surface.blit(header, (20, y))
    y += line_height
    surface.blit(font.render("-" * 30, True, (0, 0, 0)), (20, y))
    y += line_height
    # Echo lines for each word
    for raw in words:
        for offset in range(MAX_OFFSET + 1):
            line = " " * offset + raw
            txt = font.render(line, True, (0, 0, 0))
            surface.blit(txt, (20, y))
            y += line_height

render_receipt_preview(screen, sample_words, font)

# Save to an image file for quick visual check
output_path = "receipt_echo_preview.png"
pygame.image.save(screen, output_path)
print(f"Receipt preview saved to {output_path}")

# Optional: display window for interactive preview (press any key to quit)
window = pygame.display.set_mode((800, 600))
scaled = pygame.transform.smoothscale(screen, (800, 600))
window.blit(scaled, (0, 0))
pygame.display.flip()
while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT or ev.type == pygame.KEYDOWN:
            pygame.quit()
            sys.exit()

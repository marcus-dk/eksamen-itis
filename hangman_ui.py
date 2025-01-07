import pygame
import sys
from hangman import Hangman

class HangmanUI:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hangman Game")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        
        # Fonts
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 36)
        
        # Game instance
        self.game = Hangman()
        
    def draw_hangman(self):
        """Draw the hangman figure based on remaining tries."""
        # Base
        pygame.draw.line(self.screen, self.BLACK, (100, 500), (300, 500), 5)
        # Pole
        pygame.draw.line(self.screen, self.BLACK, (200, 500), (200, 100), 5)
        # Top
        pygame.draw.line(self.screen, self.BLACK, (200, 100), (400, 100), 5)
        # Rope
        pygame.draw.line(self.screen, self.BLACK, (400, 100), (400, 150), 5)
        
        tries_lost = 6 - self.game.remaining_tries
        
        if tries_lost >= 1:  # Head
            pygame.draw.circle(self.screen, self.BLACK, (400, 180), 30, 5)
        if tries_lost >= 2:  # Body
            pygame.draw.line(self.screen, self.BLACK, (400, 210), (400, 350), 5)
        if tries_lost >= 3:  # Left arm
            pygame.draw.line(self.screen, self.BLACK, (400, 250), (350, 300), 5)
        if tries_lost >= 4:  # Right arm
            pygame.draw.line(self.screen, self.BLACK, (400, 250), (450, 300), 5)
        if tries_lost >= 5:  # Left leg
            pygame.draw.line(self.screen, self.BLACK, (400, 350), (350, 450), 5)
        if tries_lost >= 6:  # Right leg
            pygame.draw.line(self.screen, self.BLACK, (400, 350), (450, 450), 5)
    
    def draw_word(self):
        """Draw the current state of the word."""
        word_surface = self.font.render(self.game.get_word_state(), True, self.BLACK)
        word_rect = word_surface.get_rect(center=(self.width // 2, 520))
        self.screen.blit(word_surface, word_rect)
    
    def draw_guessed_letters(self):
        """Draw the guessed letters."""
        guessed = "Guessed: " + " ".join(sorted(self.game.guessed_letters))
        guessed_surface = self.small_font.render(guessed, True, self.GRAY)
        self.screen.blit(guessed_surface, (20, 550))
    
    def draw_game_over(self):
        """Draw game over message."""
        if self.game.game_over:
            message = "You Won!" if self.game.won else f"Game Over! Word was: {self.game.word}"
            surface = self.font.render(message, True, self.BLACK)
            rect = surface.get_rect(center=(self.width // 2, 50))
            self.screen.blit(surface, rect)
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and not self.game.game_over:
                    if event.unicode.isalpha():
                        self.game.guess(event.unicode)
                elif event.type == pygame.KEYDOWN and self.game.game_over:
                    if event.key == pygame.K_SPACE:
                        self.game.reset_game()
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_hangman()
            self.draw_word()
            self.draw_guessed_letters()
            self.draw_game_over()
            
            if self.game.game_over:
                hint = self.small_font.render("Press SPACE to play again", True, self.GRAY)
                hint_rect = hint.get_rect(center=(self.width // 2, 450))
                self.screen.blit(hint, hint_rect)
            
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    game_ui = HangmanUI()
    game_ui.run()

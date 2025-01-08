# hangman project for ITIS class

import random
from typing import List, Set
import numpy as np

class Hangman:
    def __init__(self, word_list: np.ndarray = None, min_word_length: int = 4, max_word_length: int = 12):
        if word_list is None:
            # Read words from words.txt
            try:
                with open('words.txt', 'r') as f:
                    all_words = np.array([word.strip().lower() for word in f.readlines()])
                # Filter words by length and ensure they only contain letters
                length_mask = (np.char.str_len(all_words) >= min_word_length) & (np.char.str_len(all_words) <= max_word_length)
                alpha_mask = np.char.isalpha(all_words)
                self.word_list = all_words[length_mask & alpha_mask]
                
                if len(self.word_list) == 0:
                    raise ValueError("No words found matching the specified criteria")
            except FileNotFoundError:
                raise FileNotFoundError("words.txt file not found")
        else:
            self.word_list = word_list if isinstance(word_list, np.ndarray) else np.array(word_list)
        
        self.max_tries = 6
        self.reset_game()
    
    def reset_game(self):
        """Reset the game state."""
        self.word = np.random.choice(self.word_list).lower()
        self.guessed_letters: Set[str] = set()
        self.remaining_tries = self.max_tries
        self.game_over = False
        self.won = False
    
    def guess(self, letter: str) -> bool:
        """Make a guess. Returns True if the guess was correct."""
        if self.game_over:
            return False
        
        letter = letter.lower()
        if letter in self.guessed_letters:
            return letter in self.word
        
        self.guessed_letters.add(letter)
        
        if letter not in self.word:
            self.remaining_tries -= 1
            if self.remaining_tries <= 0:
                self.game_over = True
            return False
        
        if self._check_win():
            self.game_over = True
            self.won = True
        return True
    
    def _check_win(self) -> bool:
        """Check if all letters in the word have been guessed."""
        return all(letter in self.guessed_letters for letter in self.word)
    
    def get_word_state(self) -> str:
        """Get the current state of the word with unguessed letters as '_'."""
        return ''.join(letter if letter in self.guessed_letters else '_' for letter in self.word)
    
    def get_game_state(self):
        """Get the current game state."""
        return {
            'word_state': self.get_word_state(),
            'remaining_tries': self.remaining_tries,
            'guessed_letters': sorted(list(self.guessed_letters)),
            'game_over': self.game_over,
            'won': self.won
        }

if __name__ == "__main__":
    # Simple command-line test
    game = Hangman()
    print("Welcome to Hangman!")
    print(f"Word list size: {len(game.word_list)} words")
    
    while not game.game_over:
        state = game.get_game_state()
        print(f"\nWord: {state['word_state']}")
        print(f"Remaining tries: {state['remaining_tries']}")
        print(f"Guessed letters: {', '.join(state['guessed_letters'])}")
        
        guess = input("Enter a letter: ").strip().lower()
        if len(guess) != 1:
            print("Please enter a single letter!")
            continue
            
        correct = game.guess(guess)
        print("Correct!" if correct else "Wrong!")
    
    print("\nGame Over!")
    print(f"The word was: {game.word}")
    print("You won!" if game.won else "You lost!")

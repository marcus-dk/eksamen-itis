# hangman simple strategies
from collections import Counter
import math
from typing import List, Set, Dict
import argparse
from hangman import Hangman
import numpy as np
import csv

def calculate_letter_frequencies(words: List[str]) -> Dict[str, float]:
    """Calculate the frequency of each letter in the word list."""
    all_letters = ''.join(words).lower()
    counter = Counter(all_letters)
    total = sum(counter.values())
    return {letter: count/total for letter, count in counter.items()}

def calculate_letter_entropy(words: List[str], letter: str, pattern: str) -> float:
    """
    Calculate the information gain (entropy) for a given letter based on the current pattern.
    This considers both the presence of the letter and its positions.
    """
    if not words:
        return 0.0
    
    total_words = len(words)
    pattern_groups = {}
    
    # For each word, calculate what the pattern would be if we guessed this letter
    # This gives us the exact distribution of possible outcomes
    for word in words:
        new_pattern = list(pattern)
        for i, char in enumerate(word.lower()):
            if char == letter:
                new_pattern[i] = letter
        new_pattern = ''.join(new_pattern)
        pattern_groups[new_pattern] = pattern_groups.get(new_pattern, 0) + 1
    
    # Calculate entropy using the full probability distribution
    # Higher entropy means more information gained from guessing this letter
    entropy = 0.0
    for count in pattern_groups.values():
        p = count / total_words
        entropy -= p * math.log2(p)
    
    # Add a bonus for letters that appear in more words
    # This helps break ties in favor of more common letters
    words_with_letter = sum(1 for word in words if letter in word.lower())
    coverage = words_with_letter / total_words
    
    return entropy + (0.1 * coverage)  # Small bonus for coverage

class HangmanStrategy:
    def __init__(self, word_list: List[str]):
        self.original_words = word_list
        self.letter_frequencies = calculate_letter_frequencies(word_list)
    
    def get_next_guess(self, current_pattern: str, guessed_letters: Set[str]) -> str:
        """Abstract method to be implemented by specific strategies."""
        raise NotImplementedError
    
    def _get_possible_words(self, current_pattern: str, guessed_letters: Set[str]) -> List[str]:
        """Filter words based on current pattern and guessed letters."""
        possible_words = []
        pattern_chars = list(current_pattern.lower())
        
        for word in self.original_words:
            if len(word) != len(pattern_chars):
                continue
            
            matches = True
            word_lower = word.lower()
            
            # Check if word matches current pattern
            for i, (pattern_char, word_char) in enumerate(zip(pattern_chars, word_lower)):
                if pattern_char != '_' and pattern_char != word_char:
                    matches = False
                    break
            
            # Check if word is consistent with guessed letters
            if matches:
                for letter in guessed_letters:
                    # If letter was guessed but isn't in pattern where it should be
                    letter_in_word = letter in word_lower
                    letter_in_pattern = letter in pattern_chars
                    if letter_in_word != letter_in_pattern:
                        matches = False
                        break
            
            if matches:
                possible_words.append(word)
        
        return possible_words

class MostCommonLettersStrategy(HangmanStrategy):
    def get_next_guess(self, current_pattern: str, guessed_letters: Set[str]) -> str:
        """Return the most frequent letter from remaining possible words that hasn't been guessed yet."""
        possible_words = self._get_possible_words(current_pattern, guessed_letters)
        
        # If no possible words found (shouldn't happen in normal play), fall back to original frequencies
        if not possible_words:
            available_letters = {
                letter: freq for letter, freq in self.letter_frequencies.items()
                if letter not in guessed_letters
            }
            return max(available_letters.items(), key=lambda x: x[1])[0]
        
        # Calculate frequencies from remaining possible words
        current_frequencies = calculate_letter_frequencies(possible_words)
        
        # Filter to only unguessed letters
        available_letters = {
            letter: freq for letter, freq in current_frequencies.items()
            if letter not in guessed_letters
        }
        
        return max(available_letters.items(), key=lambda x: x[1])[0]

class LeastCommonLettersStrategy(HangmanStrategy):
    def get_next_guess(self, current_pattern: str, guessed_letters: Set[str]) -> str:
        """Return the least frequent letter from remaining possible words that hasn't been guessed yet."""
        possible_words = self._get_possible_words(current_pattern, guessed_letters)
        
        # If no possible words found (shouldn't happen in normal play), fall back to original frequencies
        if not possible_words:
            available_letters = {
                letter: freq for letter, freq in self.letter_frequencies.items()
                if letter not in guessed_letters
            }
            return min(available_letters.items(), key=lambda x: x[1])[0]
        
        # Calculate frequencies from remaining possible words
        current_frequencies = calculate_letter_frequencies(possible_words)
        
        # Filter to only unguessed letters
        available_letters = {
            letter: freq for letter, freq in current_frequencies.items()
            if letter not in guessed_letters
        }
        
        return min(available_letters.items(), key=lambda x: x[1])[0]

class EntropyStrategy(HangmanStrategy):
    def get_next_guess(self, current_pattern: str, guessed_letters: Set[str]) -> str:
        """Return the letter with highest information gain that hasn't been guessed yet."""
        possible_words = self._get_possible_words(current_pattern, guessed_letters)
        
        # If no possible words found (shouldn't happen in normal play), fall back to letter frequencies
        if not possible_words:
            available_letters = {
                letter: freq for letter, freq in self.letter_frequencies.items()
                if letter not in guessed_letters
            }
            return max(available_letters.items(), key=lambda x: x[1])[0]
        
        # Calculate entropy for each unguessed letter
        entropies = {}
        for letter in set('abcdefghijklmnopqrstuvwxyz') - guessed_letters:
            entropies[letter] = calculate_letter_entropy(possible_words, letter, current_pattern)
        
        # Return letter with highest entropy
        return max(entropies.items(), key=lambda x: x[1])[0]

def play_game(strategy: HangmanStrategy, game: Hangman, show_progress: bool = False) -> bool:
    """Play a single game using the given strategy. Returns True if won."""
    while not game.game_over:
        state = game.get_game_state()
        if show_progress:
            print(f"\nWord: {state['word_state']}")
            print(f"Remaining tries: {state['remaining_tries']}")
            print(f"Guessed letters: {', '.join(state['guessed_letters'])}")
        
        guess = strategy.get_next_guess(state['word_state'], set(state['guessed_letters']))
        if show_progress:
            print(f"Strategy guessed: {guess}")
        
        correct = game.guess(guess)
        if show_progress:
            print("Correct!" if correct else "Wrong!")
    
    if show_progress:
        print("\nGame Over!")
        print(f"The word was: {game.word}")
        print("Won!" if game.won else "Lost!")
    
    


    return game.won

def evaluate_strategy(strategy_name: str, num_games: int = 100, show_example: bool = True) -> float:
    """Evaluate a strategy over multiple games and return win rate."""
    game = Hangman()
    strategy_class = {'mc': MostCommonLettersStrategy,
        'lc': LeastCommonLettersStrategy,
        'e': EntropyStrategy
    }[strategy_name]
    
    strategy = strategy_class(game.word_list)
    wins = 0
    
    # Play one game with progress if show_example is True
    if show_example:
        print(f"\nExample game using {strategy_name} strategy:")
        game.reset_game()
        won = play_game(strategy, game, show_progress=True)
        wins += int(won)
        num_games -= 1
    
    # Play remaining games
    for _ in range(num_games):
        game.reset_game()
        won = play_game(strategy, game, show_progress=False)
        wins += int(won)
        
        # Takes the results and appends it to data_"stragety".csv, depending which stragety was used
        result = (int(game.won),game.word,len(game.word))
        data_filename = "data_" + strategy_name + ".csv"
        with open(data_filename,mode='a',newline='') as data:
            writer = csv.writer(data)
            writer.writerow(result)        
    
    win_rate = wins / (num_games + (1 if show_example else 0))
    print(f"\nStrategy: {strategy_name}")
    print(f"Games played: {num_games + (1 if show_example else 0)}")
    print(f"Win rate: {win_rate:.2%}")
    
    return win_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Hangman strategies')
    parser.add_argument('strategy', choices=['mc', 'lc', 'e'],
                      help='Which strategy to use')
    parser.add_argument('--games', type=int, default=100,
                      help='Number of games to play (default: 100)')
    parser.add_argument('--no-example', action='store_true',
                      help='Skip showing an example game')
    
    args = parser.parse_args()
    evaluate_strategy(args.strategy, args.games, not args.no_example)
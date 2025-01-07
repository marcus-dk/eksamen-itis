import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hangman import Hangman
from collections import Counter
import random
from typing import List, Tuple, Set, Dict
import os
import argparse
import re

class WordConstraintSolver:
    def __init__(self, min_word_length: int = 4, max_word_length: int = 12):
        # Load and preprocess words using numpy arrays
        with open('words.txt', 'r') as f:
            all_words = np.array([word.strip().lower() for word in f.readlines()])
        
        # Filter words by length and ensure they only contain letters
        length_mask = (np.char.str_len(all_words) >= min_word_length) & (np.char.str_len(all_words) <= max_word_length)
        alpha_mask = np.char.isalpha(all_words)
        self.word_list = all_words[length_mask & alpha_mask]
        
        # Pre-compute letter positions for all words
        self.word_lengths = np.char.str_len(self.word_list)
        self.max_length = max(self.word_lengths)
        
        # Create padded word arrays with a special padding character
        self.word_arrays = np.full((len(self.word_list), self.max_length), '_', dtype=str)
        for i, word in enumerate(self.word_list):
            self.word_arrays[i, :len(word)] = list(word)
        
        # Pre-compute letter positions using the padded arrays
        self.letter_positions = {}
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            # Create a boolean mask for each letter's positions in all words
            self.letter_positions[letter] = np.any(self.word_arrays == letter, axis=1)
        
        # Cache for word possibilities
        self.cache = {}
    
    def get_possible_words(self, word_state: str, guessed_letters: Set[str]) -> np.ndarray:
        """Return all possible words that match the current game state using vectorized operations."""
        cache_key = (word_state, tuple(sorted(guessed_letters)))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Filter by length using pre-computed lengths
        length_mask = self.word_lengths == len(word_state)
        possible_indices = np.where(length_mask)[0]
        
        if len(possible_indices) == 0:
            self.cache[cache_key] = np.array([])
            return np.array([])
        
        # Create masks for revealed positions using vectorized operations
        word_arrays = self.word_arrays[possible_indices]
        for i, char in enumerate(word_state):
            if char != '_':
                possible_indices = possible_indices[word_arrays[:, i] == char]
                if len(possible_indices) == 0:
                    self.cache[cache_key] = np.array([])
                    return np.array([])
                word_arrays = self.word_arrays[possible_indices]
        
        # Filter out words containing incorrect guesses using pre-computed letter positions
        incorrect_letters = {letter for letter in guessed_letters if letter not in word_state}
        if incorrect_letters:
            incorrect_mask = np.zeros(len(possible_indices), dtype=bool)
            for letter in incorrect_letters:
                incorrect_mask |= self.letter_positions[letter][possible_indices]
            possible_indices = possible_indices[~incorrect_mask]
        
        result = self.word_list[possible_indices]
        self.cache[cache_key] = result
        return result
    
    def get_letter_probabilities(self, possible_words: np.ndarray, guessed_letters: Set[str]) -> Dict[str, float]:
        """Calculate letter probabilities using vectorized operations."""
        if len(possible_words) == 0:
            return {chr(i + ord('a')): 0.0 for i in range(26)}
        
        # Get indices of possible words
        possible_indices = np.array([np.where(self.word_list == word)[0][0] for word in possible_words])
        word_arrays = self.word_arrays[possible_indices]
        word_count = len(possible_words)
        letter_probs = {}
        
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter in guessed_letters:
                letter_probs[letter] = 0.0
                continue
            
            # Calculate letter frequency and coverage using vectorized operations
            letter_mask = word_arrays == letter
            words_with_letter = np.sum(np.any(letter_mask, axis=1))
            total_occurrences = np.sum(letter_mask)
            
            if words_with_letter > 0:
                # Only count non-padding positions for frequency score
                valid_positions = word_arrays != '_'
                total_valid_positions = np.sum(valid_positions)
                frequency_score = total_occurrences / total_valid_positions if total_valid_positions > 0 else 0
                coverage_score = words_with_letter / word_count
                letter_probs[letter] = 0.4 * frequency_score + 0.6 * coverage_score
            else:
                letter_probs[letter] = 0.0
        
        return letter_probs

class HangmanNet(nn.Module):
    def __init__(self, input_size: int = 79, hidden_size: int = 64):  # Reduced hidden size
        super(HangmanNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 26),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class HangmanAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HangmanNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.word_solver = WordConstraintSolver()
        self.hybrid_weight = 0.7  # Adjusted weight for better balance
        
        # Pre-allocate tensors for efficiency
        self.state_vector = torch.zeros(79, device=self.device)
        self.target_vector = torch.zeros(26, device=self.device)
    
    def _create_game_state_vector(self, game: Hangman) -> torch.Tensor:
        """Convert game state to input vector with word possibility information."""
        state = game.get_game_state()
        self.state_vector.zero_()
        
        # Original features
        word_state = state['word_state']
        total_chars = len(word_state)
        char_counts = Counter(word_state)
        
        # Vectorized letter counting
        for char, count in char_counts.items():
            if char != '_':
                self.state_vector[ord(char.lower()) - ord('a')] = count / total_chars
        
        # Vectorized guessed letters
        for letter in state['guessed_letters']:
            self.state_vector[26 + ord(letter) - ord('a')] = 1
        
        self.state_vector[52] = state['remaining_tries'] / 6
        
        # Add word possibility information
        possible_words = self.word_solver.get_possible_words(word_state, set(state['guessed_letters']))
        if len(possible_words) > 0:
            word_arrays = np.array([list(word) for word in possible_words])
            letter_counts = np.sum(word_arrays == np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])[:, None, None], axis=(1, 2))
            self.state_vector[53:79] = torch.from_numpy(letter_counts / len(possible_words)).to(self.device)
        
        return self.state_vector
    
    def train(self, num_episodes: int = 1000, batch_size: int = 32):
        """Train the model using mini-batches for better efficiency."""
        self.model.train()
        game = Hangman()
        running_loss = 0
        
        for episode in range(num_episodes):
            game.reset_game()
            episode_states = []
            episode_targets = []
            max_steps = 26
            step = 0
            
            while not game.game_over and step < max_steps:
                # Collect state and target
                state_vector = self._create_game_state_vector(game)
                episode_states.append(state_vector.clone())
                
                # Create target
                self.target_vector.zero_()
                word_letters = set(game.word)
                for letter in word_letters:
                    self.target_vector[ord(letter) - ord('a')] = 1.0
                episode_targets.append(self.target_vector.clone())
                
                # Make a guess using hybrid approach
                with torch.no_grad():
                    guess = self.get_guess(game)
                game.guess(guess)
                step += 1
                
                # Train on mini-batch when enough samples are collected
                if len(episode_states) >= batch_size:
                    states_batch = torch.stack(episode_states)
                    targets_batch = torch.stack(episode_targets)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(states_batch)
                    loss = self.criterion(outputs, targets_batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    episode_states = []
                    episode_targets = []
            
            # Train on remaining samples
            if episode_states:
                states_batch = torch.stack(episode_states)
                targets_batch = torch.stack(episode_targets)
                
                self.optimizer.zero_grad()
                outputs = self.model(states_batch)
                loss = self.criterion(outputs, targets_batch)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            if episode % 100 == 0:
                avg_loss = running_loss / (episode + 1)
                print(f"Episode {episode}, Average Loss: {avg_loss:.4f}, " +
                      f"Won: {game.won}, Word: {game.word}")
                running_loss = 0
    
    def get_guess(self, game: Hangman) -> str:
        """Get the next guess using improved hybrid approach."""
        self.model.eval()
        state = game.get_game_state()
        guessed_letters = set(state['guessed_letters'])
        
        # Get word constraint predictions
        possible_words = self.word_solver.get_possible_words(state['word_state'], guessed_letters)
        constraint_probs = self.word_solver.get_letter_probabilities(possible_words, game.guessed_letters)
        
        # Get neural network predictions
        with torch.no_grad():
            state_vector = self._create_game_state_vector(game)
            nn_output = self.model(state_vector)
            nn_probs = {chr(i + ord('a')): float(nn_output[i]) for i in range(26)}
        
        # Combine predictions with dynamic weighting
        combined_probs = {}
        for letter in constraint_probs:
            if letter not in guessed_letters:
                # Adjust weight based on number of possible words
                dynamic_weight = min(0.9, self.hybrid_weight + (0.1 if len(possible_words) < 100 else 0))
                combined_probs[letter] = (
                    dynamic_weight * constraint_probs[letter] +
                    (1 - dynamic_weight) * nn_probs[letter]
                )
            else:
                combined_probs[letter] = 0.0
        
        # Return letter with highest probability
        return max(combined_probs.items(), key=lambda x: x[1])[0]
    
    def save_model(self, path: str = "hangman_model.pth"):
        """Save the trained model to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = "hangman_model.pth"):
        """Load a trained model from a file."""
        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return False
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
        return True

def evaluate_model(model: HangmanAI, num_games: int = 100) -> Tuple[float, float]:
    """Evaluate the model's performance."""
    game = Hangman()
    wins = 0
    total_tries = 0
    
    for _ in range(num_games):
        game.reset_game()
        game_tries = 0
        max_steps = 26
        
        while not game.game_over and game_tries < max_steps:
            guess = model.get_guess(game)
            game.guess(guess)
            game_tries += 1
        
        if game.won:
            wins += 1
        total_tries += game_tries
    
    win_rate = wins / num_games
    avg_tries = total_tries / num_games
    return win_rate, avg_tries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hangman AI Training and Evaluation')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--model-path', type=str, default='hangman_model.pth', help='Path to save/load model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    args = parser.parse_args()
    
    # Create AI instance
    ai = HangmanAI()
    
    if args.train:
        # Train new model
        print("Training model...")
        ai.train(num_episodes=args.episodes)
        ai.save_model(args.model_path)
    else:
        # Try to load existing model
        if not ai.load_model(args.model_path):
            print("No saved model found. Training new model...")
            ai.train(num_episodes=args.episodes)
            ai.save_model(args.model_path)
    
    print("\nEvaluating model...")
    win_rate, avg_tries = evaluate_model(ai)
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average tries per game: {avg_tries:.2f}")
    
    # Play a sample game
    print("\nPlaying a sample game...")
    game = Hangman()
    while not game.game_over:
        state = game.get_game_state()
        print(f"\nWord: {state['word_state']}")
        print(f"Remaining tries: {state['remaining_tries']}")
        print(f"Guessed letters: {', '.join(state['guessed_letters'])}")
        
        guess = ai.get_guess(game)
        print(f"AI guesses: {guess}")
        game.guess(guess)
    
    print("\nGame Over!")
    print(f"The word was: {game.word}")
    print("AI won!" if game.won else "AI lost!")

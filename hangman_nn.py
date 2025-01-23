import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from typing import List, Set, Dict, Tuple
from hangman import Hangman
import re
import os
import argparse
import csv

class HangmanNetwork(nn.Module):
    """Simple feed-forward network for Hangman letter prediction."""
    def __init__(self, input_size: int = 79):
        super(HangmanNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 26),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class CustomHangmanLoss(nn.Module):
    """Custom loss function that considers game state and winning probability."""
    def __init__(self):
        super(CustomHangmanLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                remaining_tries: torch.Tensor, word_lengths: torch.Tensor) -> torch.Tensor:
        # Base BCE loss
        bce_loss = self.bce(predictions, targets)
        
        # Scale loss based on remaining tries (higher penalty when fewer tries remain)
        tries_weight = 1.0 + (6.0 - remaining_tries) / 2.0
        
        # Scale loss based on word length (longer words need more efficient guessing)
        length_weight = word_lengths / 6.0  # normalize by average word length
        
        # Increase penalty for false positives (guessing wrong letters)
        fp_mask = (predictions > 0.5) & (targets == 0)
        fp_penalty = torch.where(fp_mask, 2.0, 1.0)
        
        # Reduce penalty for true positives (reward correct guesses)
        tp_mask = (predictions > 0.5) & (targets == 1)
        tp_reward = torch.where(tp_mask, 0.5, 1.0)
        
        # Combine all factors
        weighted_loss = bce_loss * tries_weight.unsqueeze(1) * length_weight.unsqueeze(1) * fp_penalty * tp_reward
        
        return weighted_loss.mean()

class HangmanNeuralSolver:
    """Neural network-based Hangman solver with word possibility tracking."""
    def __init__(self, word_list: List[str]):
        self.word_list = word_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HangmanNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = CustomHangmanLoss()
        
        # Cache for possible words
        self.word_cache = {}
        
        # Training history
        self.training_history = {'loss': [], 'win_rate': []}

    def _get_possible_words(self, pattern: str, guessed_letters: Set[str]) -> List[str]:
        """Get all possible words that match the current game state."""
        cache_key = (pattern, tuple(sorted(guessed_letters)))
        if cache_key in self.word_cache:
            return self.word_cache[cache_key]

        possible_words = []
        pattern_regex = pattern.replace('_', '[a-z]')
        
        for word in self.word_list:
            if len(word) != len(pattern):
                continue
            
            if not re.match(pattern_regex, word):
                continue
                
            # Check if word contains any incorrectly guessed letters
            incorrect_letters = guessed_letters - set(pattern)
            if any(letter in word for letter in incorrect_letters):
                continue
                
            possible_words.append(word)
        
        self.word_cache[cache_key] = possible_words
        return possible_words

    def _create_input_vector(self, pattern: str, guessed_letters: Set[str], remaining_tries: int) -> torch.Tensor:
        """
        Create enhanced input vector:
        - First 26: letter presence in pattern
        - Next 26: guessed letters
        - Next 26: letter frequencies in possible words
        - Last 1: remaining tries
        """
        input_vector = torch.zeros(79, device=self.device)
        
        # Pattern information (first 26)
        for letter in pattern:
            if letter != '_':
                input_vector[ord(letter) - ord('a')] = 1.0
                
        # Guessed letters (next 26)
        for letter in guessed_letters:
            input_vector[26 + ord(letter) - ord('a')] = 1.0
            
        # Letter frequencies in possible words (next 26)
        possible_words = self._get_possible_words(pattern, guessed_letters)
        if possible_words:
            # Count all letters in possible words
            letter_counts = Counter(''.join(possible_words))
            total_letters = sum(letter_counts.values())
            
            # Normalize frequencies
            if total_letters > 0:
                for letter, count in letter_counts.items():
                    input_vector[52 + ord(letter) - ord('a')] = count / total_letters
        
        # Remaining tries (last 1)
        input_vector[78] = remaining_tries / 6  # Normalize by max tries
            
        return input_vector

    def _create_target_vector(self, word: str) -> torch.Tensor:
        """Create target vector indicating correct letters."""
        target = torch.zeros(26, device=self.device)
        for letter in word.lower():
            target[ord(letter) - ord('a')] = 1.0
        return target

    def train(self, num_episodes: int = 1000, batch_size: int = 32):
        """Train using episodes with improved state representation and batch training."""
        self.model.train()
        game = Hangman()
        running_loss = 0.0
        num_batches = 0
        
        print("Training neural network...")
        for episode in range(num_episodes):
            game.reset_game()
            episode_states = []
            episode_targets = []
            remaining_tries_batch = []
            word_lengths_batch = []
            game_moves = []
            
            # Play full game and collect experiences
            while not game.game_over:
                state = game.get_game_state()
                input_vector = self._create_input_vector(
                    state['word_state'],
                    set(state['guessed_letters']),
                    state['remaining_tries']
                )
                
                with torch.no_grad():
                    output = self.model(input_vector)
                    guess = self._get_guess_from_output(output, state['word_state'], set(state['guessed_letters']))
                
                # Store state before making move
                game_moves.append({
                    'input_vector': input_vector,
                    'target_vector': self._create_target_vector(game.word),
                    'remaining_tries': state['remaining_tries'],
                    'word_length': len(game.word),
                    'guess': guess,
                    'state_before': state.copy()
                })
                
                game.guess(guess)
            
            # Calculate rewards based on game outcome
            game_won = game.won
            
            # Process moves with outcome-based rewards
            for move in game_moves:
                reward_scale = 2.0 if game_won else 0.5  # Simpler reward scaling
                
                episode_states.append(move['input_vector'])
                episode_targets.append(move['target_vector'] * reward_scale)
                remaining_tries_batch.append(move['remaining_tries'])
                word_lengths_batch.append(move['word_length'])
                
                # Update model when we have enough samples
                if len(episode_states) >= batch_size:
                    self.optimizer.zero_grad()
                    states_batch = torch.stack(episode_states)
                    targets_batch = torch.stack(episode_targets)
                    tries_batch = torch.tensor(remaining_tries_batch, device=self.device, dtype=torch.float32)
                    lengths_batch = torch.tensor(word_lengths_batch, device=self.device, dtype=torch.float32)
                    
                    outputs = self.model(states_batch)
                    loss = self.criterion(outputs, targets_batch, tries_batch, lengths_batch)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    num_batches += 1
                    
                    episode_states = []
                    episode_targets = []
                    remaining_tries_batch = []
                    word_lengths_batch = []
            
            if episode % 100 == 0:
                avg_loss = running_loss / max(num_batches, 1)  # Avoid division by zero
                self.training_history['loss'].append(avg_loss)
                
                win_rate, _ = self.evaluate(50)
                self.training_history['win_rate'].append(win_rate)
                
                print(f"Episode {episode}, Loss: {avg_loss:.4f}, " +
                      f"Win Rate: {win_rate:.2%}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                running_loss = 0.0
                num_batches = 0
                
                if win_rate > 0.95:
                    print("Reached high win rate, stopping training.")
                    break

    def _get_guess_from_output(self, output: torch.Tensor, pattern: str, guessed_letters: Set[str]) -> str:
        """Convert network output to letter guess using possible words information."""
        # Get network probabilities
        probabilities = output.detach().cpu().numpy()
        

            
        # Zero out already guessed letters
        for letter in guessed_letters:
            probabilities[ord(letter) - ord('a')] = 0
            
        # Get possible words and their letter frequencies
        possible_words = self._get_possible_words(pattern, guessed_letters)
        if possible_words:
            # Count letters in possible words
            letter_counts = Counter(''.join(possible_words))
            
            # Boost probabilities for letters that appear in possible words
            for letter, count in letter_counts.items():
                if letter not in guessed_letters:
                    idx = ord(letter) - ord('a')
                    probabilities[idx] *= (1 + count / len(possible_words))
            
            # Extra boost for most common letters in possible words
            if len(possible_words) < 10:  # When we have few possibilities, trust the word list more
                most_common = Counter(''.join(possible_words)).most_common(3)
                for letter, _ in most_common:
                    if letter not in guessed_letters:
                        probabilities[ord(letter) - ord('a')] *= 1.5
        
        # Return letter with highest adjusted probability
        return chr(ord('a') + np.argmax(probabilities))

    def get_next_guess(self, pattern: str, guessed_letters: Set[str], remaining_tries: int) -> str:
        """Get next guess using the trained neural network and word possibilities."""
        self.model.eval()
        with torch.no_grad():
            input_vector = self._create_input_vector(pattern, guessed_letters, remaining_tries)
            output = self.model(input_vector)
            return self._get_guess_from_output(output, pattern, guessed_letters)

    def save_model(self, path: str = "hangman_model.pth"):
        """Save the trained model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = "hangman_model.pth"):
        """Load a trained model and optimizer state."""
        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return False
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from {path}")
        return True

    def evaluate(self, num_games: int = 100) -> Tuple[float, float]:
        """Evaluate the model's performance."""
        self.model.eval()
        game = Hangman()
        wins = 0
        total_tries = 0
        
        for _ in range(num_games):
            game.reset_game()
            tries = 0
            
            while not game.game_over and tries < 26:
                state = game.get_game_state()
                guess = self.get_next_guess(
                    state['word_state'],
                    set(state['guessed_letters']),
                    state['remaining_tries']
                )
                game.guess(guess)
                tries += 1
            
            if game.won:
                wins += 1
            total_tries += tries
            
            # Collect data in same format as other strategies
            result = (int(game.won), game.word, len(game.word),(6-game.remaining_tries))
            with open('data_nn.csv', mode='a', newline='') as data_file:
                writer = csv.writer(data_file)
                writer.writerow(result)
        
        return wins / num_games, total_tries / num_games

def main():
    parser = argparse.ArgumentParser(description='Hangman Neural Network Solver')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--model-path', type=str, default='hangman_model.pth', help='Path to save/load model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval-games', type=int, default=100, help='Number of evaluation games')
    args = parser.parse_args()
    
    # Create solver
    game = Hangman()
    solver = HangmanNeuralSolver(game.word_list)
    
    if args.train:
        # Train new model
        print("Training new model...")
        solver.train(num_episodes=args.episodes)
        solver.save_model(args.model_path)
    else:
        # Try to load existing model
        if not solver.load_model(args.model_path):
            print("No saved model found. Training new model...")
            solver.train(num_episodes=args.episodes)
            solver.save_model(args.model_path)
    
    # Evaluate
    print("\nEvaluating model...")
    win_rate, avg_tries = solver.evaluate(args.eval_games)
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average tries per game: {avg_tries:.2f}")
    
    # Play sample game
    print("\nPlaying a sample game...")
    game.reset_game()
    target_word = game.word
    guesses = []
    
    while not game.game_over:
        state = game.get_game_state()
        print(f"\nWord: {state['word_state']}")
        print(f"Remaining tries: {state['remaining_tries']}")
        print(f"Guessed letters: {', '.join(sorted(state['guessed_letters']))}")
        
        guess = solver.get_next_guess(
            state['word_state'],
            set(state['guessed_letters']),
            state['remaining_tries']
        )
        correct = game.guess(guess)
        guesses.append((guess, correct))
        print(f"Guessed: {guess} ({'✓' if correct else '✗'})")
    
    print("\nGame Over!")
    print(f"Word was: {target_word}")
    print("Result: " + ("Won! 🎉" if game.won else "Lost 😢"))
    print("Guesses: " + ' '.join(f"{g}{'✓' if c else '✗'}" for g, c in guesses))

if __name__ == "__main__":
    main()

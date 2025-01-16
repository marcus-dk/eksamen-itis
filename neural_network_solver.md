# Neural Network Hangman Solver

## Overview
The Neural Network Hangman Solver implements a deep learning approach to playing Hangman using PyTorch. The system combines traditional game state tracking with a neural network that learns optimal letter selection strategies through experience and reinforcement.

## Architecture

```mermaid
classDiagram
    class HangmanNetwork {
        +forward(x)
        -network: Sequential
    }
    
    class CustomHangmanLoss {
        +forward(predictions, targets, remaining_tries, word_lengths)
        -bce: BCELoss
    }
    
    class HangmanNeuralSolver {
        +word_list: List[str]
        +model: HangmanNetwork
        +optimizer: Adam
        +criterion: CustomHangmanLoss
        +word_cache: Dict
        +training_history: Dict
        +train()
        +get_next_guess()
        +evaluate()
    }
    
    HangmanNeuralSolver --> HangmanNetwork
    HangmanNeuralSolver --> CustomHangmanLoss
```

## Neural Network Architecture

### Input Layer (79 neurons)
```mermaid
flowchart TD
    A[Input Vector] --> B[Letter Presence<br/>26 neurons]
    A --> C[Guessed Letters<br/>26 neurons]
    A --> D[Letter Frequencies<br/>26 neurons]
    A --> E[Remaining Tries<br/>1 neuron]
    
    B --> F[Hidden Layer 1<br/>128 neurons]
    C --> F
    D --> F
    E --> F
```

1. **Pattern Information** (26 neurons)
   - Binary encoding of revealed letters
   - 1.0 for revealed letters in pattern
   - 0.0 for unrevealed positions

2. **Guessed Letters** (26 neurons)
   - Binary encoding of all attempted letters
   - 1.0 for previously guessed letters
   - 0.0 for available letters

3. **Letter Frequencies** (26 neurons)
   - Normalized frequencies from possible words
   - Higher values for common letters
   - Derived from remaining word possibilities

4. **Game State** (1 neuron)
   - Normalized remaining tries (tries/6)
   - Provides urgency information to network

### Hidden Layers
```mermaid
flowchart LR
    A[Input<br/>79] --> B[Linear + ReLU<br/>128]
    B --> C[Linear + ReLU<br/>64]
    C --> D[Linear + Sigmoid<br/>26]
    D --> E[Output<br/>Letter Probabilities]
```

1. **First Hidden Layer** (128 neurons)
   - Linear transformation with ReLU activation
   - Learns complex letter patterns
   - Maps game state to feature space

2. **Second Hidden Layer** (64 neurons)
   - Reduced dimensionality for focus
   - ReLU activation for non-linearity
   - Refines feature representations

3. **Output Layer** (26 neurons)
   - Sigmoid activation for probabilities
   - One neuron per possible letter
   - Values represent guess confidence

## Custom Loss Function

```mermaid
flowchart TD
    A[Predictions] --> E[BCE Loss]
    B[Targets] --> E
    C[Remaining Tries] --> F[Tries Weight]
    D[Word Length] --> G[Length Weight]
    
    E --> H[Base Loss]
    F --> I[Weighted Loss]
    G --> I
    
    subgraph Penalty Adjustments
        J[False Positive<br/>Penalty x2.0] --> K[Final Loss]
        L[True Positive<br/>Reward x0.5] --> K
    end
    
    H --> K
    I --> K
```

### Components
1. **Base Loss**
   - Binary Cross-Entropy (BCE)
   - Measures prediction accuracy
   - Formula: -Σ(y·log(p) + (1-y)·log(1-p))

2. **Dynamic Weighting**
   ```python
   tries_weight = 1.0 + (6.0 - remaining_tries) / 2.0
   length_weight = word_lengths / 6.0
   ```
   - Increases penalty for fewer tries
   - Scales with word complexity

3. **Strategic Penalties**
   - False Positive Penalty (2.0x)
   - True Positive Reward (0.5x)
   - Encourages conservative guessing

## Training Process

```mermaid
flowchart TD
    A[Initialize Game] --> B[Get Game State]
    B --> C[Generate Input Vector]
    C --> D[Network Prediction]
    D --> E[Make Guess]
    E --> F[Update Game State]
    F --> G{Game Over?}
    G -->|No| B
    G -->|Yes| H[Calculate Rewards]
    H --> I[Backpropagate]
    I --> J[Update Weights]
    
    subgraph Batch Processing
        K[Collect Experiences] --> L[Process Batch]
        L --> M[Optimize Network]
    end
```

### Training Strategy
1. **Episode Collection**
   - Full games are played
   - States and actions stored
   - Outcomes determine rewards

2. **Batch Processing**
   - 32 moves per batch
   - Gradient clipping at 1.0
   - Adam optimizer (lr=0.001)

3. **Reward Structure**
   ```python
   reward_scale = 2.0 if game_won else 0.5
   ```
   - Winning games heavily rewarded
   - Losing games penalized

## Word Possibility Tracking

```mermaid
flowchart LR
    A[Game State] --> B[Pattern Matching]
    B --> C[Letter Filtering]
    C --> D[Cache Key Generation]
    D --> E{Cache Hit?}
    E -->|Yes| F[Return Cached]
    E -->|No| G[Calculate New]
    G --> H[Update Cache]
    H --> I[Return Results]
```

### Caching System
1. **Cache Key**
   - Pattern string
   - Sorted guessed letters
   - Ensures consistent lookup

2. **Pattern Matching**
   - Regex-based filtering
   - Length verification
   - Letter consistency check

3. **Performance Impact**
   - Reduces computation
   - Speeds up training
   - Memory-efficient storage

## Guess Selection Logic

```mermaid
flowchart TD
    A[Network Output] --> B[Get Probabilities]
    B --> C[Zero Guessed Letters]
    
    subgraph Word Analysis
        D[Get Possible Words] --> E[Count Letters]
        E --> F[Boost Probabilities]
    end
    
    C --> G[Apply Word Analysis]
    F --> G
    G --> H[Select Best Letter]
```

### Selection Process
1. **Network Probabilities**
   - Raw sigmoid outputs
   - 26 letter confidences
   - Range: [0.0, 1.0]

2. **Probability Adjustment**
   ```python
   probabilities[idx] *= (1 + count / len(possible_words))
   ```
   - Based on word possibilities
   - Frequency boosting
   - Position awareness

3. **Final Selection**
   - Highest adjusted probability
   - Filtered for validity
   - Strategic weighting

## Performance Metrics

### Evaluation
- Win rate tracking
- Average tries per game
- Training loss history
- Strategy effectiveness

### Model Persistence
- State dict saving
- Optimizer state backup
- Training history storage
- Easy model reloading

## Usage

```bash
# Train new model
python hangman_nn.py --train --episodes 1000

# Evaluate existing model
python hangman_nn.py --eval-games 100

# Custom model path
python hangman_nn.py --model-path "custom_model.pth"
```

## Implementation Details
- PyTorch framework
- CUDA/CPU compatibility
- Efficient state representation
- Sophisticated loss function
- Strategic reward shaping
``` 
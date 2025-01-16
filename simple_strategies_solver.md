# Simple Strategies Hangman Solver

## Overview
The Simple Strategies Hangman Solver implements three different algorithmic approaches to playing Hangman, each using statistical and information theory concepts without neural networks. The strategies are based on letter frequency analysis and information gain calculations.

## Architecture

```mermaid
classDiagram
    class HangmanStrategy {
        <<abstract>>
        +word_list: List[str]
        +letter_frequencies: Dict[str, float]
        +get_next_guess()*
        +_get_possible_words()
    }
    
    class MostCommonLettersStrategy {
        +get_next_guess()
    }
    
    class LeastCommonLettersStrategy {
        +get_next_guess()
    }
    
    class EntropyStrategy {
        +get_next_guess()
    }
    
    HangmanStrategy <|-- MostCommonLettersStrategy
    HangmanStrategy <|-- LeastCommonLettersStrategy
    HangmanStrategy <|-- EntropyStrategy
```

## Strategies

### 1. Most Common Letters Strategy

```mermaid
flowchart TD
    A[Current Pattern] --> B[Get Possible Words]
    B --> C[Calculate Letter Frequencies]
    D[Previously Guessed Letters] --> E[Filter Available Letters]
    C --> E
    E --> F[Select Most Frequent Letter]
```

- Uses letter frequency analysis from remaining possible words
- Falls back to original word list frequencies if no possible words found
- Selects the most frequently occurring unguessed letter

### 2. Least Common Letters Strategy

```mermaid
flowchart TD
    A[Current Pattern] --> B[Get Possible Words]
    B --> C[Calculate Letter Frequencies]
    D[Previously Guessed Letters] --> E[Filter Available Letters]
    C --> E
    E --> F[Select Least Frequent Letter]
```

- Inverse of the Most Common Letters Strategy
- Useful for words containing rare letters
- Helps eliminate unlikely letters quickly

### 3. Entropy Strategy

```mermaid
flowchart TD
    A[Current Pattern] --> B[Get Possible Words]
    B --> C[Calculate Information Gain]
    D[Previously Guessed Letters] --> E[Filter Letters]
    C --> E
    E --> F[Select Highest Entropy Letter]
    
    subgraph Information Gain Calculation
        G[Pattern Distribution] --> H[Entropy Calculation]
        H --> I[Coverage Bonus]
        I --> C
    end
```

- Based on information theory principles
- Calculates information gain (entropy) for each possible guess
- Considers both letter presence and positions
- Includes coverage bonus for common letters

## Core Components

### 1. Word Filtering System

```mermaid
flowchart LR
    A[Input Pattern] --> D[Length Filter]
    B[Guessed Letters] --> E[Pattern Matcher]
    C[Word List] --> D
    D --> E
    E --> F[Letter Consistency Check]
    F --> G[Possible Words]
```

- Filters words based on:
  - Length matching
  - Pattern compatibility
  - Guessed letter consistency
- Maintains game state integrity

### 2. Letter Frequency Analysis

```mermaid
graph TD
    A[Word List] --> B[Join All Words]
    B --> C[Count Letters]
    C --> D[Calculate Frequencies]
    D --> E[Normalize Counts]
```

- Calculates letter frequencies from word list
- Normalizes frequencies for fair comparison
- Used by both Common Letters strategies

### 3. Entropy Calculation

```mermaid
flowchart TD
    A[Words] --> B[Pattern Groups]
    B --> C[Probability Distribution]
    C --> D[Shannon Entropy]
    D --> F[Final Score]
    E[Coverage Bonus] --> F
```

- Implements Shannon entropy calculation
- Groups words by resulting patterns
- Adds coverage bonus for practical effectiveness
- Formula: Entropy = -Σ(p * log2(p)) + 0.1 * coverage

## Usage

```bash
python hangman_ss.py mc  # Use Most Common Letters Strategy
python hangman_ss.py lc  # Use Least Common Letters Strategy
python hangman_ss.py e   # Use Entropy Strategy
```

Additional options:
- `--games N`: Number of games to play
- `--no-example`: Skip example game display

## Performance Analysis

Each strategy's performance is tracked and saved:
- Win/loss ratio
- Word length correlation
- Success rate by word
- Results saved to strategy-specific CSV files

## Strategy Selection Guide

1. **Most Common Letters Strategy**
   - Best for: Common words with typical letter distributions
   - Advantages: Simple, fast, generally effective
   - Disadvantages: Struggles with unusual words

2. **Least Common Letters Strategy**
   - Best for: Words with rare letters
   - Advantages: Can find unusual patterns quickly
   - Disadvantages: Less effective for common words

3. **Entropy Strategy**
   - Best for: General purpose use
   - Advantages: Balanced approach, considers position information
   - Disadvantages: More computationally intensive

## Implementation Details

- Pure Python implementation
- No external dependencies except NumPy
- Efficient caching of word possibilities
- Strategy-specific result tracking
- Command-line interface for easy testing 
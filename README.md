# eksamen-itis
project for exam in Introduction to Intelligent System - DTU

## project
the basis of the project is a hangman game.

the end product for the exam is a lab report comparing different strategies for solving the hangman game.

the idea is to do statistical analysis of the different strategies and compare them to each other.

the strategies are:

- most common letters
- least common letters
- entropy of the letters
- strategy resulting from machine learning

### goals
to have a hangman game that can be played with and without ui (ui for people who want to play the game and see how they stack up against the different strategies / terminal for training the machine learning model without ui)

to have a machine learning model that can be trained to play the game

to collect data on the different strategies and compare them to each other

## structure

- hangman.py: the hangman game
- hangman_ui.py: the hangman game with a ui
- hangman_ml.py: file for training the machine learning model
- data.csv: data collected from the different strategies
- data_analysis.py: the data analysis of the different strategies


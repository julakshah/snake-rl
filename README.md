# snake-rl

Machine Learning final project - reinforcement learning for the Snake game.

## Requirements

## Setup

1. Clone the github repository
2. Install requirements by running `pip install -r requirements.txt` in project

I'd recommend creating a python venv at this point for all the packages being created

## Running

There are a two entrypoints into the program. You can either run train.py which
will begin training a new model immediately. Alternatively, you can run
model_play.py which will load the current model and see it play snake.

1. Navigate to the repository root folder.
2. Make sure all setup instructions are followed and env is activate (if you made one).
3. Follow Training The Model or Running The Model command

### Training The Model

```bash
python3 main.py
```

### Running The Model

```bash
python3 model_play.py
```

## Acknowledgements

- This project relies heavily on lychanl's (Gymnasium_Snake_Game)[https://github.com/lychanl/Gymnasium_Snake_Game] package. Huge thank you to lychanl for making this!
- This project heavily referenced
  - GeeksForGeeks' (page on DQL reinforcement learning)[https://www.geeksforgeeks.org/deep-learning/deep-q-learning/]
  - Adam Paszke's and Mark Towers' (post on DQN reinforcement learning)[https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html]
  - Jamesnorthfield's (Medium post on DQN's)[https://medium.com/@jamesnorthfield2001/a-guide-to-deep-q-networks-dqns-806f6f4805f4]
- Claude Sonnet 4.6 was used for debugging

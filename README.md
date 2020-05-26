# Doom-AI
Deep Reinforcement Learning applied to the game DOOM, using the game environment *VizDoom*.

## Dependencies
* Python 3.6
* VizDoom
* Tensorflow 2.1

## How to run
Adjust parameters on the end of the document and then:
```
python3 doom-ai.py
```
To get real-time tracking of the training just access the logs produced by Tensorboard itself:
```
tensorboard --logdir logs/dqn/folder_with_current_time_name
```

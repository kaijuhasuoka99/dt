# Simple Decision Transformer Implementation

## Description
This is a code implementation based on the [Decision Transformer](https://arxiv.org/abs/2106.01345).

I collected demonstration data for Atari Breakout using a model trained with [PPO]().

The Decision Transformer is a reinforcement learning method that conditions actions based on rewards to achieve those rewards, but the improvement in accuracy through this method has not been thoroughly demonstrated.

In that regard, there are several examples of imitation learning without using rewards, implemented with a similar architecture, such as [RT-1](https://robotics-transformer1.github.io/) and [Gato](https://arxiv.org/abs/2205.06175).

Instead of using rewards, learning from high-quality demonstration data yields better results.

Therefore, the implementation does not use reward-based conditioning. It relies purely on states and actions.

The demonstration data only includes episodes with a score of 400 or higher from PPO. As a result, the final average score was 396!

Note that, due to the small scale of the implementation, the Transformer has been set with 4 heads and 4 layers each.

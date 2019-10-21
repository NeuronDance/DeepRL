

# QMIX

## Theory

 论文中提出了一种能以**中心化**的端到端的方式训练去中心化策略的基于价值的全新方法 QMIX。QMIX 能够将仅基于局部观察的每个智能体的价值以复杂的非线性方式组合起来，估计**联合的动作-价值**。 

## Algorithm

## ![img](D:\Github\DeepRL_skylark\DRL-Multi-Agent\QMIX.assets\v2-98cea01bf7d7d2239d4d50460a57e6cf_hd.jpg)![img](D:\Github\DeepRL_skylark\DRL-Multi-Agent\QMIX.assets\v2-79fe8838e84d6def61e3db6cf7332428_hd.jpg)

## Paper

 **QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning** [ ICML 2018 ](https://arxiv.org/pdf/1803.11485.pdf ) 

## Application

论文中假设环境为模拟器环境或实验室环境， 在这种环境下，**交流的限制被解除**，**全局信息**是可以获得的。 

本文 在**starCraft**游戏上实验 

## Code

**QMIX by Ray framework**:  https://github.com/ray-project/ray/tree/master/rllib/agents/qmix 



## Cite






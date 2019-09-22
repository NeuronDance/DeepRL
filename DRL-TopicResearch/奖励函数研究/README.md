# Gym环境奖励函数研究

本项目为开源公益项目，致力于研究强化学习中的奖励函数工程，研究主题为Gym环境中例子。

### 环境简介
OpenAI Gym是一款用于研发和比较强化学习算法的工具包，它支持训练智能体（agent）做任何事，是一个用于开发和比较RL 算法的工具包，与其他的数值计算库兼容，如tensorflow 或者theano 库。现在主要支持的是python 语言，以后将支持其他语言。官方提供的gym文档。

OpenAI Gym包含两部分：

> gym 开源 包含一个测试问题集，每个问题成为环境（environment），可以用于自己的强化学习算法开发，这些环境有共享的接口，允许用户设计通用的算法，例如：Atari、CartPole等。

> OpenAI Gym 服务
提供一个站点和api ，允许用户对自己训练的算法进行性能比较。

Gym环境整体分为一下：
+ Classic control and toy text:
  提供了一些RL相关论文中的一些小问题，开始学习Gym从这开始！
+ Algorithmic:
提供了学习算法的环境，比如翻转序列这样的问题，虽然能很容易用直接编程实现，但是单纯用例子来训练RL模型有难度的。这些问题有一个很好的特性： 能够通过改变序列长度改变难度。
+ Atari:
这里提供了一些小游戏，比如我们小时候玩过的小蜜蜂，弹珠等等。这些问题对RL研究有着很大影响！
+ Board games:
提供了Go这样一个简单的下棋游戏，由于这个问题是多人游戏，Gym提供有opponent与你训练的agent进行对抗。
+ 2D and 3D robots:
机器人控制环境。 这些问题用 MuJoCo 作为物理引擎。

### 研究环境

Gym 环境是深度强化学习发展的重要基础，相关实例包括一下环境，更多请查阅Gym官网
1. Acrobot-v1
2. CartPole-v1
3. Pendulum-v0
4. MountainCar-v0
5. MountainCarContinous-v0
6. Pong-v0

7. Ant-v0
8. Halfcheetah-v2
9. Hopper-v2
10. InvertedDoublePendulum-v2
11. InvertedPendulum-v2
12. Reacher-v2
13. Swimmer-v2
14. Walker2d-v2

15. FetchPickAndPlace-v0
16. FetchPush-v0
17. FechReach-v0
18. FetchSlide-v0


### 初期研究流程和步骤
整体学习以文档形式展现，格式如下：

（1）克隆仓库，并在environment目录下创建对应环境的文件，例如：Pendulum-v0.md

（2）在Pendulum-v0.md中
>
    + 介绍环境，包括原理图，github对应链接等
    + 介绍奖励函数的设置过程和设置原理
    + 对gym环境中涉及奖励的源码进行讲解

 (3)总结

实例请看environment目录下：Pendulum-v0.md

### 致谢列表：
非常感谢以下同学对Gym环境解读工作所做的贡献：

>
> @GithubAccountName 贡献： Pendulum-v0贡献

> @ChasingZenith   贡献：Ant-v0环境

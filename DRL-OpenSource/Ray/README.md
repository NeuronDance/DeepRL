# Ray

Ray 是由UC Berkeley 推出的高性能分布式框架，其中包括如下三种用于机器学习的加速库：

- Tune：可扩展的超参数调整库（[文档](https://ray.readthedocs.io/en/latest/tune.html)）
- RLlib：可扩展的强化学习库（[文档](https://ray.readthedocs.io/en/latest/rllib.html)）
- [Distributed Training](https://ray.readthedocs.io/en/latest/distributed_training.html)

Ray文档见[链接](https://ray.readthedocs.io/en/latest/)

## Ray 安装

### 最新版安装

```
pip install -U ray  # also recommended: ray[debug]
```

### 源码安装

#### 依赖库

**Ubuntu:**

```
sudo apt-get update
sudo apt-get install -y build-essential curl unzip psmisc

# If you are not using Anaconda, you need the following.
sudo apt-get install python-dev  # For Python 2.
sudo apt-get install python3-dev  # For Python 3.

pip install cython==0.29.0
```

**MacOS:**

```
brew update
brew install wget

pip install cython==0.29.0
```

**Anaconda:**

```
conda install libgcc
```

#### Ray 安装

```
git clone https://github.com/ray-project/ray.git

# Install Bazel.
ray/ci/travis/install-bazel.sh

# Optionally build the dashboard (requires Node.js, see below for more information).
pushd ray/python/ray/dashboard/client
npm ci
npm run build
popd

# Install Ray.
cd ray/python
pip install -e . --verbose  # Add --user if you see a permission denied error.
```

## RLlib

### Algorithms

先介绍一下RLlib，RLlib中提供了几乎所有state-of-the-art 的DRL算法的实现，包括tensorflow和pytorch

- High-throughput architectures:  Ape-X | IMPALA | APPO
- Gradient-based:  (A2C, A3C) | (DDPG, TD3) |(DQN, Rainbow, Parametric DQN) | Policy Gradients | PPO | SAC
- Derivative-free:  ARS | Evolution Strategies |  (QMIX, VDN, IQN) | MADDPG | MARWIL

RLlib Algorithms 详见 [链接](https://ray.readthedocs.io/en/latest/rllib-algorithms.html#)

### 使用方式

1. 直接使用 ALG_NAME.Trainer，以PPO举例：

   ```python
   import ray
   import ray.rllib.agents.ppo as ppo
   from ray.tune.logger import pretty_print
   
   ray.init() 
   config = ppo.DEFAULT_CONFIG.copy()
   config["num_gpus"] = 0
   config["num_workers"] = 1
   config["eager"] = False
   trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
   
   # Can optionally call trainer.restore(path) to load a checkpoint.
   
   for i in range(1000):
      # Perform one iteration of training the policy with PPO
      result = trainer.train()
      print(pretty_print(result))
   
      if i % 100 == 0:
          checkpoint = trainer.save()
          print("checkpoint saved at", checkpoint)
   ```

2. 使用 **Tune** （官方推荐）， 同样是PPO：

   ```python
   import ray
   from ray import tune
   
   ray.init()
   tune.run(
       "PPO",
       stop={"episode_reward_mean": 200},
       config={
           "env": "CartPole-v0",
           "num_gpus": 0,
           "num_workers": 1,
           "lr": tune.grid_search([0.01, 0.001, 0.0001]),
           "eager": False,
       },
   )
   ```

### Tensorboard支持

Ray 会**自动**在根目录创建一个叫做ray_results的文件夹并保存checkpoint，直接在终端中运行：

```
tensorboard --logdir ~/ray_results
```

即可打开tensorboard查看结果

### 可扩展性

RLlib提供了自定义培训的几乎所有方面的方法，包括环境，神经网络模型，动作分布和策略定义：

![_images/rllib-components.svg](D:\Github\DeepRL\DRL-OpenSource\Ray\README.assets\rllib-components.svg)

## Tune

TODO


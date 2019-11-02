## A Guide Resource for Deep Reinforcement Learning

---
Contents
----
- __[Books](#Books)__
- __[Courses ](#Courses)__
- __[Environment](#Environment)__
- __[Algorithm](#Algorithms)__
- __[OpenSourceFramework](#OpenSourceFramework)__
- __[Talks / Lectures](#talks)__





- __[Guides](#guides)__
- __[Guides](#guides)__

- __[Papers](#papers)__
- __[Blog Posts](#blog-posts)__
- __[Video examples](#video-examples)__
- __[Code Examples](#code-examples)__
- __[Datasets](#datasets)__
- __[Frameworks](#frameworks)__
- __[Contributing](#Cite)__

---

Books
-----
1. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf) by Richard S. Sutton and Andrew G. Barto (2017),[Chinese-Edtion](https://rl.qiwihui.com/zh_CN/latest/index.html), [Code](http://www.incompleteideas.net/book/code/code2nd.html)
2. [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) by Csaba Szepesvari (updated 2019)
3. [Deep Reinforcement Learning Hands-On]((https://www.amazon.com/Deep-Reinforcement-Learning-Hands-Q-networks/dp/1788834240)) by Maxim Lapan (2018),[Code](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)
4. [Reinforcement learning, State-Of-The- Art](https://github.com/yangmingustb/planning_books_1/blob/master/Reinforcement%20learning%20state%20of%20the%20art.pdf) by Marco Wiering, Martijin van Otterlo
5. [Deep Reinforcement Learning in Action](https://www.manning.com/books/deep-reinforcement-learning-in-action) by Alexander Zai and Brandon Brown (in progress)
6. [Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning) by Miguel Morales (in progress)
7. [Multi-Agent Machine Learning A Reinforcement Approach【百度云链接】]() by Howard M.Schwartz(2017)
8. [强化学习在阿里的技术演进与业务创新](https://alitech-private.oss-cn-beijing.aliyuncs.com/1517812754285/reinforcement_learning.pdf?Expires=1572504433&OSSAccessKeyId=LTAIqKGWQyF6Vd3W&Signature=gqCg4wSejW2sqWNluFDQebMWA94%3D) by Alibaba Group
9. [Hands-On Reinforcement Learning with Python(百度云链接)]()
10. [Reinforcement Learning And Optimal Control](http://web.mit.edu/dimitrib/www/RLbook.html) by Dimitri P. Bertsekas, 2019

> Note：Some Chinese books for the purpose of making money are not recommended here.

Courses
----
1. [UCL Course on RL(★★★)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) by David Sliver, [Video-en](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ),[Video-zh](https://www.bilibili.com/video/av51567407/)
2. [OpenAI's Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) by OpenAI(2018)
3. [Stanford CS-234: Reinforcement Learning](http://web.stanford.edu/class/cs234/index.html) (2019), [Videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)

4. [DeepMind Advanced Deep Learning & Reinforcement Learning](http://www.cs.ucl.ac.uk/current_students/syllabus/undergrad/compmi22_advanced_deep_learning_and_reinforcement_learning/) (2018),[Videos](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)

5. [GeorgiaTech CS-8803 Deep Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600) (2018?)
6. [UC Berkeley CS294-112 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) (2018 Fall),[Video-zh](https://www.bilibili.com/video/av39816961)
7. [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) by  Berkeley CA(2017)
8. [Thomas Simonini's Deep Reinforcement Learning Course](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
9. [CS-6101 Deep Reinforcement Learning ](https://www.comp.nus.edu.sg/~kanmy/courses/6101_1820/), NUS SoC, 2018/2019, Semester II
10. [Course on Reinforcement Learning](http://researchers.lille.inria.fr/~lazaric/Webpage/MVA-RL_Course17.html) by Alessandro Lazaric，2018


Survey and
----
1. [Deep Reinforcement Learning](https://arxiv.org/pdf/1810.06339.pdf) by Yuxi Li(2018)
2. sheji
3. ge


[Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) by Irpan, Alex(2018), [Read Chinese](https://zhuanlan.zhihu.com/p/34089913)

Environment
----
普通环境

- https://gym.openai.com/envs/#classic_control
- https://github.com/erlerobot/gym-gazebo
- https://github.com/robotology/gym-ignition
- https://github.com/dartsim/gym-dart
- https://github.com/Roboy/gym-roboy
- https://github.com/openai/retro
- https://github.com/openai/gym-soccer
- https://github.com/duckietown/gym-duckietown
- Unity: https://github.com/Unity-Technologies/ml-agents (multiagent)
- https://github.com/koulanurag/ma-gym (multiagent)
- https://github.com/ucuapps/modelicagym
- https://github.com/mwydmuch/ViZDoom
- https://github.com/benelot/pybullet-gym
- https://github.com/Healthcare-Robotics/assistive-gym
- https://github.com/Microsoft/malmo
- https://github.com/nadavbh12/Retro-Learning-Environment
- https://github.com/twitter/torch-twrl

- https://github.com/arex18/rocket-lander
- https://github.com/ppaquette/gym-doom
- https://github.com/thedimlebowski/Trading-Gym
- More: https://github.com/Phylliade/awesome-openai-gym-environments
- https://github.com/deepmind/pysc2 (by DeepMind) (Blizzard StarCraft II Learning Environment (SC2LE) component)




Baselines and Benchmarks
---
1. https://github.com/openai/baselines [【stalbe-baseline】](https://github.com/hill-a/stable-baselines)
2. [rl-baselines-zoo](https://github.com/araffin/rl-baselines-zoo)
3. [ROBEL](https://sites.google.com/view/roboticsbenchmarks) ([google-research/robel](https://github.com/google-research/robel/))
4. [RLBench](https://sites.google.com/view/rlbench) ([stepjam/RLBench](https://github.com/stepjam/RLBench))
5. https://martin-thoma.com/sota/#reinforcment-learning
6. https://github.com/rlworkgroup/garage
7. [Atari Environments Scores](https://www.endtoend.ai/envs/gym/atari/)


Algorithms
----




Frameworks
----

1. [OpenAI Gym](https://gym.openai.com) ([GitHub](https://github.com/openai/gym)) ([docs](https://gym.openai.com/docs/))

2. rllab ([GitHub](https://github.com/rll/rllab)) ([readthedocs](http://rllab.readthedocs.io))
3. Ray [(Doc)](https://ray.readthedocs.io/en/latest/index.html)
4. Dopamine: https://github.com/google/dopamine (uses some tensorflow)
5. trfl: https://github.com/deepmind/trfl (uses tensorflow)
6.  ChainerRL ([GitHub](https://github.com/chainer/chainerrl)) (API: Python)
7. Surreal [GitHub](https://github.com/SurrealAI/surreal)  (API: Python) (support: Stanford Vision and Learning Lab).[Paper](https://surreal.stanford.edu/img/surreal-corl2018.pdf)
8. PyMARL [GitHub](https://github.com/oxwhirl/pymarl) (support: http://whirl.cs.ox.ac.uk/)
9. TF-Agents: https://github.com/tensorflow/agents (uses tensorflow)
10. TensorForce ([GitHub](https://github.com/reinforceio/tensorforce)) (uses tensorflow)
11. [RL-Glue](https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue) ([Google Code Archive](https://code.google.com/archive/p/rl-glue-ext/wikis/RLGlueCore.wiki)) (API: C/C++, Java, Matlab, Python, Lisp) (support: Alberta)
12. MAgent https://github.com/geek-ai/MAgent (uses tensorflow)
13. RLlib http://ray.readthedocs.io/en/latest/rllib.html (API: Python)
14. http://burlap.cs.brown.edu/ (API: Java)
15. [rlpyt: A Research Code Base for Deep Reinforcement Learning in PyTorch](https://bair.berkeley.edu/blog/2019/09/24/rlpyt/)



Applications
----
1. [Reinforcement Learning Applications](https://medium.com/@yuxili/rl-applications-73ef685c07eb)
2.

Guides
-----

- Related awesome guides
  - [awesome-rl #1](https://github.com/aikorea/awesome-rl)
  - [awesome-rl #2](https://aikorea.org/awesome-rl/)
  - [awesome-deep-rl](https://github.com/tigerneil/awesome-deep-rl): Inspiring but too focused on specific implementations
  - [awesome-rl-nlp](https://github.com/adityathakker/awesome-rl-nlp): NLP for the win
- Complete guides
  - [A Beginner's Guide to Deep Reinforcement Learning by skymind](https://skymind.ai/wiki/deep-reinforcement-learning)
  - http://www.argmin.net/2018/06/25/outsider-rl/
  - https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
- Short introductions
  - A Medium Series
    - https://medium.com/@m.alzantot/deep-reinforcement-learning-demystified-episode-0-2198c05a6124
    - https://becominghuman.ai/genetic-algorithm-for-reinforcement-learning-a38a5612c4dc
    - https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
- https://github.com/andri27-ts/60_Days_RL_Challenge
- https://github.com/jachiam/rl-intro
- http://planspace.org/20170830-berkeley_deep_rl_bootcamp/
- https://github.com/google/dopamine

Talks
----

- Introduction to Deep Reinforcement Learning by Lex Fridman (2019)
    - [slides](https://www.dropbox.com/s/wekmlv45omd266o/deep_rl_intro.pdf?dl=0)
    - [video](https://www.youtube.com/watch?v=zR11FLZ-O9M)
- Reinforcement Learning for the People and/or by the People by Emma Brunskill (2017)
    - [slides](https://cs.stanford.edu/people/ebrun/NIPS_2017_tutorial_brunskill.pdf)
    - [video](https://www.facebook.com/nipsfoundation/videos/1555771847847382/)
- [Deep Reinforcement Learning](https://www.youtube.com/watch?v=PtAIh9KSnjo) by John Schulman (2016)
- [Deep Robotic Learning](https://www.youtube.com/watch?v=eKaYnXQUb2g) by Sergey Levine (2017)

Papers
----

OpenAI's [Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)

Survey:
  - [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1806.08894) (2018)
  - [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866) (2017)

Results:
  - Learning to Optimize Join Queries With Deep Reinforcement Learning
    - [Technical paper](https://arxiv.org/abs/1808.03196)
    - [Technical blog post](https://databeta.wordpress.com/2018/09/20/the-crossroads-of-ai-and-database-algorithms-query-optimization/)
    - [Nontechnical blog post])(https://databeta.wordpress.com/2018/09/20/the-crossroads-of-ai-and-database-algorithms-query-optimization/)

Blog Posts
----

- [Deep Reinforcement Learning — Policy Gradients — Lunar Lander!](https://medium.com/@gabogarza/deep-reinforcement-learning-policy-gradients-8f6df70404e6)
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) by Andrej Karpathy

Video examples
------

- [DL Agent learning to play Atari](https://www.youtube.com/watch?v=MKtNv1UOaZA)
- [Pong, Tic Tac Toe, and Super Mario with Deep Q-learning](https://www.youtube.com/watch?v=scLTbin8FiQ)
- [Miscellaneous robots learning](https://www.youtube.com/playlist?list=PL5nBAYUyJTrM48dViibyi68urttMlUv7e)


Code Examples
------

- [OpenAI Baselines](https://github.com/openai/baselines): A set of high-quality implementations of reinforcement learning algorithms
- [Mingo](https://github.com/tensorflow/minigo): An open-source implementation of the AlphaGoZero algorithm
- Adventures with Sutton and Barto
  - [Python implementation](https://github.com/levimcclenny/Reinforcement_Learning)
  - Rendered Jupyter Notebooks
    - [MDP](http://people.tamu.edu/~levimcclenny/project/reinforcement-learning/Barto_Sutton_RL/Finite_MDPs/)
    - [DP](http://people.tamu.edu/~levimcclenny/project/reinforcement-learning/Barto_Sutton_RL/Dynamic_Programming/)


Datasets
----

- [MLPerf's benchmark suite for RL](https://github.com/mlperf/training/tree/master/reinforcement)


Topics
----
##### Rewards
1. [Deep Reinforcement Learning Models: Tips & Tricks for Writing Reward Functions](https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0)
2. [Meta Reward Learning](https://towardsdatascience.com/introducing-meta-reward-learning-5916ea2766de)

##### Policy Gradient
1. [Policy Gradient](https://github.com/reinforcement-learning-kr/pg_travel)
2.

##### Distributed Training Reinforcement Learning
1. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) by ICML 2016.paper
2. [GA3C: GPU-based A3C for Deep Reinforcement Learning](https://www.researchgate.net/publication/310610848_GA3C_GPU-based_A3C_for_Deep_Reinforcement_Learning) by Iuri Frosio, Stephen Tyree, NIPS 2016
3. [Distributed Prioritized Experience Replay](https://openreview.net/pdf?id=H1Dy---0Z) by Dan Horgan, John Quan, David Budden,ICLR 2018
4. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) by Lasse Espeholt, Hubert Soyer, Remi Munos ,ICML 2018
5. [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/pdf/1804.08617.pdf) by Gabriel Barth-Maron, Matthew W. Hoffman, ICLR 2018.

6. [Emergence of Locomotion Behaviours in Rich Environments](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1707.02286) by Nicolas Heess, Dhruva TB, Srinivasan Sriram, 2017

7. [GPU-Accelerated Robotic Simulation for Distributed Reinforcement Learning](https://arxiv.org/abs/1810.05762) by Jacky Liang, Viktor Makoviychuk, 2018

8. [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/pdf?id=r1lyTjAqYX) bySteven Kapturowski, Georg Ostrovski,  ICLR 2019.


##### Model-free RL
1. **playing atari with deep reinforcement learning** NIPS Deep Learning Workshop 2013. [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

    *Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller*

2. **Human-level control through deep reinforcement learning** Nature 2015. [paper](https://www.nature.com/articles/nature14236)

    *Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis*

3. **Deep Reinforcement Learning with Double Q-learning** AAAI 16. [paper](https://arxiv.org/abs/1509.06461)

    *Hado van Hasselt, Arthur Guez, David Silver*

4. **Dueling Network Architectures for Deep Reinforcement Learning** ICML16. [paper](https://arxiv.org/abs/1511.06581)

    *Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas*

5. **Deep Recurrent Q-Learning for Partially Observable MDPs** AAA15. [paper](https://arxiv.org/abs/1507.06527)

    *Matthew Hausknecht, Peter Stone*

6. **Prioritized Experience Replay** ICLR 2016. [paper](https://arxiv.org/abs/1507.06527)

    *Tom Schaul, John Quan, Ioannis Antonoglou, David Silver*

7. **Asynchronous Methods for Deep Reinforcement Learning** ICML2016. [paper](https://arxiv.org/abs/1602.01783)

    *Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu*

8. **A Distributional Perspective on Reinforcement Learning** ICML2017. [paper](https://arxiv.org/abs/1707.06887)

    *Marc G. Bellemare, Will Dabney, Rémi Munos*

9. **Noisy Networks for Exploration** ICLR2018. [paper](https://arxiv.org/abs/1706.10295)

    *Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, Shane Legg*

10. **Rainbow: Combining Improvements in Deep Reinforcement Learning** AAAI2018. [paper](https://arxiv.org/abs/1710.02298)

    *Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver*

##### Model-based RL
1. **Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion** NIPS2018. [paper](https://arxiv.org/abs/1807.01675)

    *Jacob Buckman, Danijar Hafner, George Tucker, Eugene Brevdo, Honglak Lee*

2. **Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning**  ICML2018.[paper](https://arxiv.org/abs/1803.00101)

    *Vladimir Feinberg, Alvin Wan, Ion Stoica, Michael I. Jordan, Joseph E. Gonzalez, Sergey Levine*

3. **Value Prediction Network** NIPS2017. [paper](https://arxiv.org/abs/1707.03497)

    *Vladimir Feinberg, Alvin Wan, Ion Stoica, Michael I. Jordan, Joseph E. Gonzalez, Sergey Levine*

4. **Imagination-Augmented Agents for Deep Reinforcement Learning** NIPS2017. [paper](https://arxiv.org/abs/1707.06203)

    *Théophane Weber, Sébastien Racanière, David P. Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adria Puigdomènech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, Razvan Pascanu, Peter Battaglia, Demis Hassabis, David Silver, Daan Wierstra*

5. **Continuous Deep Q-Learning with Model-based Acceleration** ICML2016. [paper](https://arxiv.org/abs/1603.00748)

    *Shixiang Gu, Timothy Lillicrap, Ilya Sutskever, Sergey Levine*

6. **Uncertainty-driven Imagination for Continuous Deep Reinforcement Learning** CoRL2017. [paper](http://proceedings.mlr.press/v78/kalweit17a/kalweit17a.pdf)

    *Gabriel Kalweit, Joschka Boedecker*

7. **Model-Ensemble Trust-Region Policy Optimization** ICLR2018. [paper](https://arxiv.org/abs/1802.10592)

    *Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, Pieter Abbeel*

8. **Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models** NIPS2018. [paper](https://arxiv.org/abs/1805.12114)

    *Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine*

9. **Dyna, an integrated architecture for learning, planning, and reacting** ACM1991. [paper](https://dl.acm.org/citation.cfm?id=122377)

    *Sutton, Richard S*

10. **Learning Continuous Control Policies by Stochastic Value Gradients** NIPS 2015. [paper](https://arxiv.org/abs/1510.09142)

    *Nicolas Heess, Greg Wayne, David Silver, Timothy Lillicrap, Yuval Tassa, Tom Erez*

11. **Imagination-Augmented Agents for Deep Reinforcement Learning** NIPS 2017. [paper](https://arxiv.org/abs/1707.06203)

    *Théophane Weber, Sébastien Racanière, David P. Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adria Puigdomènech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, Razvan Pascanu, Peter Battaglia, Demis Hassabis, David Silver, Daan Wierstra*

12. **Learning and Policy Search in Stochastic Dynamical Systems with Bayesian Neural Networks** ICLR 2017. [paper](https://arxiv.org/abs/1605.07127)

    *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft*



Applications
----
1. [IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control](https://faculty.ist.psu.edu/jessieli/Publications/2018-KDD-IntelliLight.pdf) by Hua Wei，Guanjie Zheng(2018)







AlphaGoZero
----
1. https://www.depthfirstlearning.com/2018/AlphaGoZero




Multi-Agents
----

## Tutorial and Books
* [Deep Multi-Agent Reinforcement Learning](https://ora.ox.ac.uk/objects/uuid:a55621b3-53c0-4e1b-ad1c-92438b57ffa4) by Jakob N Foerster, 2018. PhD Thesis.
* [Multi-Agent Machine Learning: A Reinforcement Approach](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118884614) by H. M. Schwartz, 2014.
* [Multiagent Reinforcement Learning](http://www.ecmlpkdd2013.org/wp-content/uploads/2013/09/Multiagent-Reinforcement-Learning.pdf) by Daan Bloembergen, Daniel Hennes, Michael Kaisers, Peter Vrancx. ECML, 2013.
* [Multiagent systems: Algorithmic, game-theoretic, and logical foundations](http://www.masfoundations.org/download.html) by Shoham Y, Leyton-Brown K. Cambridge University Press, 2008.

## Review Papers
* [A Survey on Transfer Learning for Multiagent Reinforcement Learning Systems](https://www.jair.org/index.php/jair/article/view/11396) by Silva, Felipe Leno da; Costa, Anna Helena Reali. JAIR, 2019.
* [Autonomously Reusing Knowledge in Multiagent Reinforcement Learning](https://www.ijcai.org/proceedings/2018/774) by Silva, Felipe Leno da; Taylor, Matthew E.; Costa, Anna Helena Reali. IJCAI, 2018.
* [Deep Reinforcement Learning Variants of Multi-Agent Learning Algorithms](https://project-archive.inf.ed.ac.uk/msc/20162091/msc_proj.pdf) by Castaneda A O. 2016.
* [Evolutionary Dynamics of Multi-Agent Learning: A Survey](https://www.jair.org/index.php/jair/article/view/10952) by Bloembergen, Daan, et al. JAIR, 2015.
* [Game theory and multi-agent reinforcement learning](https://www.researchgate.net/publication/269100101_Game_Theory_and_Multi-agent_Reinforcement_Learning) by Nowé A, Vrancx P, De Hauwere Y M. Reinforcement Learning. Springer Berlin Heidelberg, 2012.
* [Multi-agent reinforcement learning: An overview](http://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/10_003.pdf) by Buşoniu L, Babuška R, De Schutter B. Innovations in multi-agent systems and applications-1. Springer Berlin Heidelberg, 2010
* [A comprehensive survey of multi-agent reinforcement learning](http://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/07_019.pdf) by Busoniu L, Babuska R, De Schutter B. IEEE Transactions on Systems Man and Cybernetics Part C Applications and Reviews, 2008
* [If multi-agent learning is the answer, what is the question?](http://robotics.stanford.edu/~shoham/www%20papers/LearningInMAS.pdf) by Shoham Y, Powers R, Grenager T. Artificial Intelligence, 2007.
* [From single-agent to multi-agent reinforcement learning: Foundational concepts and methods](http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/learningNeto05.pdf) by Neto G. Learning theory course, 2005.
* [Evolutionary game theory and multi-agent reinforcement learning](https://pdfs.semanticscholar.org/bb9f/bee22eae2b47bbf304804a6ac07def1aecdb.pdf) by Tuyls K, Nowé A. The Knowledge Engineering Review, 2005.
* [An Overview of Cooperative and Competitive Multiagent Learning](https://www.researchgate.net/publication/221622801_An_Overview_of_Cooperative_and_Competitive_Multiagent_Learning) by Pieter Jan ’t HoenKarl TuylsLiviu PanaitSean LukeJ. A. La Poutré. AAMAS's workshop LAMAS, 2005.
* [Cooperative multi-agent learning: the state of the art](https://cs.gmu.edu/~eclab/papers/panait05cooperative.pdf) by Liviu Panait and Sean Luke, 2005.

## Research Papers

### Framework
* [Mean Field Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1802.05438.pdf) by Yaodong Yang, Rui Luo, Minne Li, Ming Zhou, Weinan Zhang, and Jun Wang. ICML 2018.
* [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf) by Lowe R, Wu Y, Tamar A, et al. arXiv, 2017.
* [Deep Decentralized Multi-task Multi-Agent RL under Partial Observability](https://arxiv.org/pdf/1703.06182.pdf) by Omidshafiei S, Pazis J, Amato C, et al. arXiv, 2017.
* [Multiagent Bidirectionally-Coordinated Nets for Learning to Play StarCraft Combat Games](https://arxiv.org/pdf/1703.10069.pdf) by Peng P, Yuan Q, Wen Y, et al. arXiv, 2017.
* [Robust Adversarial Reinforcement Learning](https://arxiv.org/pdf/1703.02702.pdf) by Lerrel Pinto, James Davidson, Rahul Sukthankar, Abhinav Gupta. arXiv, 2017.
* [Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1702.08887.pdf) by Foerster J, Nardelli N, Farquhar G, et al. arXiv, 2017.
* [Multiagent reinforcement learning with sparse interactions by negotiation and knowledge transfer](https://arxiv.org/pdf/1508.05328.pdf) by Zhou L, Yang P, Chen C, et al. IEEE transactions on cybernetics, 2016.
* [Decentralised multi-agent reinforcement learning for dynamic and uncertain environments](https://arxiv.org/pdf/1409.4561.pdf) by Marinescu A, Dusparic I, Taylor A, et al. arXiv, 2014.
* [CLEANing the reward: counterfactual actions to remove exploratory action noise in multiagent learning](http://irll.eecs.wsu.edu/wp-content/papercite-data/pdf/2014iat-holmesparker.pdf) by HolmesParker C, Taylor M E, Agogino A, et al. AAMAS, 2014.
* [Bayesian reinforcement learning for multiagent systems with state uncertainty](http://www.fransoliehoek.net/docs/Amato13MSDM.pdf) by Amato C, Oliehoek F A. MSDM Workshop, 2013.
* [Multiagent learning: Basics, challenges, and prospects](http://www.weiss-gerhard.info/publications/AI_MAGAZINE_2012_TuylsWeiss.pdf) by Tuyls, Karl, and Gerhard Weiss. AI Magazine, 2012.
* [Classes of multiagent q-learning dynamics with epsilon-greedy exploration](http://icml2010.haifa.il.ibm.com/papers/191.pdf) by Wunder M, Littman M L, Babes M. ICML, 2010.
* [Conditional random fields for multi-agent reinforcement learning](http://www.machinelearning.org/proceedings/icml2007/papers/89.pdf) by Zhang X, Aberdeen D, Vishwanathan S V N. ICML, 2007.
* [Multi-agent reinforcement learning using strategies and voting](http://ama.imag.fr/~partalas/partalasmarl.pdf) by Partalas, Ioannis, Ioannis Feneris, and Ioannis Vlahavas. ICTAI, 2007.
* [A reinforcement learning scheme for a partially-observable multi-agent game](https://pdfs.semanticscholar.org/57fb/ae00e17c0d798559ebab0e8f4267e032f41d.pdf) by Ishii S, Fujita H, Mitsutake M, et al. Machine Learning, 2005.
* [Asymmetric multiagent reinforcement learning](http://lib.tkk.fi/Diss/2004/isbn9512273594/article1.pdf) by Könönen V. Web Intelligence and Agent Systems, 2004.
* [Adaptive policy gradient in multiagent learning](http://dl.acm.org/citation.cfm?id=860686) by Banerjee B, Peng J. AAMAS, 2003.
* [Reinforcement learning to play an optimal Nash equilibrium in team Markov games](https://papers.nips.cc/paper/2171-reinforcement-learning-to-play-an-optimal-nash-equilibrium-in-team-markov-games.pdf) by Wang X, Sandholm T. NIPS, 2002.
* [Multiagent learning using a variable learning rate](https://www.sciencedirect.com/science/article/pii/S0004370202001212) by Michael Bowling and Manuela Veloso, 2002.
* [Value-function reinforcement learning in Markov game](http://www.sts.rpi.edu/~rsun/si-mal/article3.pdf) by Littman M L. Cognitive Systems Research, 2001.
* [Hierarchical multi-agent reinforcement learning](http://researchers.lille.inria.fr/~ghavamza/my_website/Publications_files/agents01.pdf) by Makar, Rajbala, Sridhar Mahadevan, and Mohammad Ghavamzadeh. The fifth international conference on Autonomous agents, 2001.
* [An analysis of stochastic game theory for multiagent reinforcement learning](https://www.cs.cmu.edu/~mmv/papers/00TR-mike.pdf) by Michael Bowling and Manuela Veloso, 2000.

### Joint action learning
* [AWESOME: A general multiagent learning algorithm that converges in self-play and learns a best response against stationary opponents](http://www.cs.cmu.edu/~conitzer/awesomeML06.pdf) by Conitzer V, Sandholm T. Machine Learning, 2007.
* [Extending Q-Learning to General Adaptive Multi-Agent Systems](https://papers.nips.cc/paper/2503-extending-q-learning-to-general-adaptive-multi-agent-systems.pdf) by Tesauro, Gerald. NIPS, 2003.
* [Multiagent reinforcement learning: theoretical framework and an algorithm.](http://www.lirmm.fr/~jq/Cours/3cycle/module/HuWellman98icml.pdf) by Hu, Junling, and Michael P. Wellman. ICML, 1998.
* [The dynamics of reinforcement learning in cooperative multiagent systems](http://www.aaai.org/Papers/AAAI/1998/AAAI98-106.pdf) by Claus C, Boutilier C. AAAI, 1998.
* [Markov games as a framework for multi-agent reinforcement learning](https://www.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf) by Littman, Michael L. ICML, 1994.

### Cooperation and competition
* [Emergent complexity through multi-agent competition](https://arxiv.org/pdf/1710.03748.pdf) by Trapit Bansal, Jakub Pachocki, Szymon Sidor, Ilya Sutskever, Igor Mordatch, 2018.
* [Learning with opponent learning awareness](https://arxiv.org/pdf/1709.04326.pdf) by Jakob Foerster, Richard Y. Chen2, Maruan Al-Shedivat, Shimon Whiteson, Pieter Abbeel, Igor Mordatch, 2018.
* [Multi-agent Reinforcement Learning in Sequential Social Dilemmas](https://arxiv.org/pdf/1702.03037.pdf) by Leibo J Z, Zambaldi V, Lanctot M, et al. arXiv, 2017. [[Post](https://deepmind.com/blog/understanding-agent-cooperation/)]
* [Reinforcement Learning in Partially Observable Multiagent Settings: Monte Carlo Exploring Policies with PAC Bounds](http://orca.st.usm.edu/~banerjee/papers/p530-ceren.pdf) by Roi Ceren, Prashant Doshi, and Bikramjit Banerjee, pp. 530-538, AAMAS 2016.
* [Opponent Modeling in Deep Reinforcement Learning](http://www.umiacs.umd.edu/~hal/docs/daume16opponent.pdf) by He H, Boyd-Graber J, Kwok K, et al. ICML, 2016.
* [Multiagent cooperation and competition with deep reinforcement learning](https://arxiv.org/pdf/1511.08779.pdf) by Tampuu A, Matiisen T, Kodelja D, et al. arXiv, 2015.
* [Emotional multiagent reinforcement learning in social dilemmas](http://www.uow.edu.au/~fren/documents/EMR_2013.pdf) by Yu C, Zhang M, Ren F. International Conference on Principles and Practice of Multi-Agent Systems, 2013.
* [Multi-agent reinforcement learning in common interest and fixed sum stochastic games: An experimental study](http://www.jmlr.org/papers/volume9/bab08a/bab08a.pdf) by Bab, Avraham, and Ronen I. Brafman. Journal of Machine Learning Research, 2008.
* [Combining policy search with planning in multi-agent cooperation](https://pdfs.semanticscholar.org/5120/d9f2c738ad223e9f8f14cb3fd5612239a35c.pdf) by Ma J, Cameron S. Robot Soccer World Cup, 2008.
* [Collaborative multiagent reinforcement learning by payoff propagation](http://www.jmlr.org/papers/volume7/kok06a/kok06a.pdf) by Kok J R, Vlassis N. JMLR, 2006.
* [Learning to cooperate in multi-agent social dilemmas](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.335&rep=rep1&type=pdf) by de Cote E M, Lazaric A, Restelli M. AAMAS, 2006.
* [Learning to compete, compromise, and cooperate in repeated general-sum games](http://www.machinelearning.org/proceedings/icml2005/papers/021_Learning_CrandallGoodrich.pdf) by Crandall J W, Goodrich M A. ICML, 2005.
* [Sparse cooperative Q-learning](http://www.machinelearning.org/proceedings/icml2004/papers/267.pdf) by Kok J R, Vlassis N. ICML, 2004.

### Coordination
* [Coordinated Multi-Agent Imitation Learning](https://arxiv.org/pdf/1703.03121.pdf) by Le H M, Yue Y, Carr P. arXiv, 2017.
* [Reinforcement social learning of coordination in networked cooperative multiagent systems](http://mipc.inf.ed.ac.uk/2014/papers/mipc2014_hao_etal.pdf) by Hao J, Huang D, Cai Y, et al. AAAI Workshop, 2014.
* [Coordinating multi-agent reinforcement learning with limited communication](http://www.aamas-conference.org/Proceedings/aamas2013/docs/p1101.pdf) by Zhang, Chongjie, and Victor Lesser. AAMAS, 2013.
* [Coordination guided reinforcement learning](http://www.ifaamas.org/Proceedings/aamas2012/papers/1B_1.pdf) by Lau Q P, Lee M L, Hsu W. AAMAS, 2012.
* [Coordination in multiagent reinforcement learning: a Bayesian approach](https://www.cs.toronto.edu/~cebly/Papers/bayesMARL.pdf) by Chalkiadakis G, Boutilier C. AAMAS, 2003.
* [Coordinated reinforcement learning](https://users.cs.duke.edu/~parr/icml02.pdf) by Guestrin C, Lagoudakis M, Parr R. ICML, 2002.
* [Reinforcement learning of coordination in cooperative multi-agent systems](http://www.aaai.org/Papers/AAAI/2002/AAAI02-050.pdf) by Kapetanakis S, Kudenko D. AAAI/IAAI, 2002.

### Security
* [Markov Security Games: Learning in Spatial Security Problems](http://www.fransoliehoek.net/docs/Klima16LICMAS.pdf) by Klima R, Tuyls K, Oliehoek F. The Learning, Inference and Control of Multi-Agent Systems at NIPS, 2016.
* [Cooperative Capture by Multi-Agent using Reinforcement Learning, Application for Security Patrol Systems](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7244682) by Yasuyuki S, Hirofumi O, Tadashi M, et al. Control Conference (ASCC), 2015
* [Improving learning and adaptation in security games by exploiting information asymmetry](http://www4.ncsu.edu/~hdai/infocom-2015-XH.pdf) by He X, Dai H, Ning P. INFOCOM, 2015.

### Self-Play
* [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/pdf/1711.00832.pdf) by Marc Lanctot, Vinicius Zambaldi, Audrunas Gruslys, Angeliki Lazaridou, Karl Tuyls, Julien Perolat, David Silver, Thore Graepel. NIPS 2017.
* [Deep reinforcement learning from self-play in imperfect-information games](https://arxiv.org/pdf/1603.01121.pdf) by Heinrich, Johannes, and David Silver. arXiv, 2016.
* [Fictitious Self-Play in Extensive-Form Games](http://jmlr.org/proceedings/papers/v37/heinrich15.pdf) by Heinrich, Johannes, Marc Lanctot, and David Silver. ICML, 2015.

### Learning To Communicate
* [Emergent Communication through Negotiation](https://openreview.net/pdf?id=Hk6WhagRW) by Kris Cao, Angeliki Lazaridou, Marc Lanctot, Joel Z Leibo, Karl Tuyls, Stephen Clark, 2018.
* [Emergence of Linguistic Communication From Referential Games with Symbolic and Pixel Input](https://openreview.net/pdf?id=HJGv1Z-AW) by Angeliki Lazaridou, Karl Moritz Hermann, Karl Tuyls, Stephen Clark
* [EMERGENCE OF LANGUAGE WITH MULTI-AGENT GAMES: LEARNING TO COMMUNICATE WITH SEQUENCES OF SYMBOLS](https://openreview.net/pdf?id=SkaxnKEYg) by Serhii Havrylov, Ivan Titov. ICLR Workshop, 2017.
* [Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning](https://arxiv.org/pdf/1703.06585.pdf) by Abhishek Das, Satwik Kottur, et al. arXiv, 2017.
* [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/pdf/1703.04908.pdf) by Igor Mordatch, Pieter Abbeel. arXiv, 2017. [[Post](https://openai.com/blog/learning-to-communicate/)]
* [Cooperation and communication in multiagent deep reinforcement learning](https://repositories.lib.utexas.edu/handle/2152/45681) by Hausknecht M J. 2017.
* [Multi-agent cooperation and the emergence of (natural) language](https://openreview.net/pdf?id=Hk8N3Sclg) by Lazaridou A, Peysakhovich A, Baroni M. arXiv, 2016.
* [Learning to communicate to solve riddles with deep distributed recurrent q-networks](https://arxiv.org/pdf/1602.02672.pdf) by Foerster J N, Assael Y M, de Freitas N, et al. arXiv, 2016.
* [Learning to communicate with deep multi-agent reinforcement learning](https://arxiv.org/pdf/1605.06676.pdf) by Foerster J, Assael Y M, de Freitas N, et al. NIPS, 2016.
* [Learning multiagent communication with backpropagation](http://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf) by Sukhbaatar S, Fergus R. NIPS, 2016.
* [Efficient distributed reinforcement learning through agreement](http://people.csail.mit.edu/lpk/papers/dars08.pdf) by Varshavskaya P, Kaelbling L P, Rus D. Distributed Autonomous Robotic Systems, 2009.

### Transfer Learning
* [Simultaneously Learning and Advising in Multiagent Reinforcement Learning](http://www.ifaamas.org/Proceedings/aamas2017/pdfs/p1100.pdf) by Silva, Felipe Leno da; Glatt, Ruben; and Costa, Anna Helena Reali. AAMAS, 2017.
* [Accelerating Multiagent Reinforcement Learning through Transfer Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14217/14005) by Silva, Felipe Leno da; and Costa, Anna Helena Reali. AAAI, 2017.
* [Accelerating multi-agent reinforcement learning with dynamic co-learning](https://web.cs.umass.edu/publication/docs/2015/UM-CS-2015-004.pdf) by Garant D, da Silva B C, Lesser V, et al. Technical report, 2015
* [Transfer learning in multi-agent systems through parallel transfer](https://www.scss.tcd.ie/~tayloral/res/papers/Taylor_ParallelTransferLearning_ICML_2013.pdf) by Taylor, Adam, et al. ICML, 2013.
* [Transfer learning in multi-agent reinforcement learning domains](https://ewrl.files.wordpress.com/2011/08/ewrl2011_submission_19.pdf) by Boutsioukis, Georgios, Ioannis Partalas, and Ioannis Vlahavas. European Workshop on Reinforcement Learning, 2011.
* [Transfer Learning for Multi-agent Coordination](https://ai.vub.ac.be/~ydehauwe/publications/ICAART2011_2.pdf) by Vrancx, Peter, Yann-Michaël De Hauwere, and Ann Nowé. ICAART, 2011.

### Imitation and Inverse Reinforcement Learning
* [Multi-Agent Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1907.13220) by Lantao Yu, Jiaming Song, Stefano Ermon. ICML 2019.
* [Multi-Agent Generative Adversarial Imitation Learning](https://papers.nips.cc/paper/7975-multi-agent-generative-adversarial-imitation-learning) by Jiaming Song, Hongyu Ren, Dorsa Sadigh, Stefano Ermon. NeurIPS 2018.
* [Cooperative inverse reinforcement learning](http://papers.nips.cc/paper/6420-cooperative-inverse-reinforcement-learning.pdf) by Hadfield-Menell D, Russell S J, Abbeel P, et al. NIPS, 2016.
* [Comparison of Multi-agent and Single-agent Inverse Learning on a Simulated Soccer Example](https://arxiv.org/pdf/1403.6822.pdf) by Lin X, Beling P A, Cogill R. arXiv, 2014.
* [Multi-agent inverse reinforcement learning for zero-sum games](https://arxiv.org/pdf/1403.6508.pdf) by Lin X, Beling P A, Cogill R. arXiv, 2014.
* [Multi-robot inverse reinforcement learning under occlusion with interactions](http://aamas2014.lip6.fr/proceedings/aamas/p173.pdf) by Bogert K, Doshi P. AAMAS, 2014.
* [Multi-agent inverse reinforcement learning](http://homes.soic.indiana.edu/natarasr/Papers/mairl.pdf) by Natarajan S, Kunapuli G, Judah K, et al. ICMLA, 2010.

### Meta Learning
* [Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments](https://arxiv.org/pdf/1710.03641.pdf) by l-Shedivat, M. 2018.


### Application
* [MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence](https://arxiv.org/pdf/1712.00600.pdf) by Zheng L et al. NIPS 2017 & AAAI 2018 Demo. ([Github Page](https://github.com/geek-ai/MAgent))
* [Collaborative Deep Reinforcement Learning for Joint Object Search](https://arxiv.org/pdf/1702.05573.pdf) by Kong X, Xin B, Wang Y, et al. arXiv, 2017.
* [Multi-Agent Stochastic Simulation of Occupants for Building Simulation](http://www.ibpsa.org/proceedings/BS2017/BS2017_051.pdf) by Chapman J, Siebers P, Darren R. Building Simulation, 2017.
* [Extending No-MASS: Multi-Agent Stochastic Simulation for Demand Response of residential appliances](http://www.ibpsa.org/proceedings/BS2017/BS2017_056.pdf) by Sancho-Tomás A, Chapman J, Sumner M, Darren R. Building Simulation, 2017.
* [Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving](https://arxiv.org/pdf/1610.03295.pdf) by Shalev-Shwartz S, Shammah S, Shashua A. arXiv, 2016.
* [Applying multi-agent reinforcement learning to watershed management](https://www.researchgate.net/profile/Karl_Mason/publication/299416955_Applying_Multi-Agent_Reinforcement_Learning_to_Watershed_Management/links/56f545b908ae95e8b6d1d3ff.pdf) by Mason, Karl, et al. Proceedings of the Adaptive and Learning Agents workshop at AAMAS, 2016.
* [Crowd Simulation Via Multi-Agent Reinforcement Learning](http://www.aaai.org/ocs/index.php/AIIDE/AIIDE10/paper/viewFile/2112/2550) by Torrey L. AAAI, 2010.
* [Traffic light control by multiagent reinforcement learning systems](https://pdfs.semanticscholar.org/61bc/b98b7ae3df894f4f72aba3d145bd48ca2cd5.pdf) by Bakker, Bram, et al. Interactive Collaborative Information Systems, 2010.
* [Multiagent reinforcement learning for urban traffic control using coordination graphs](https://staff.science.uva.nl/s.a.whiteson/pubs/kuyerecml08.pdf) by Kuyer, Lior, et al. oint European Conference on Machine Learning and Knowledge Discovery in Databases, 2008.
* [A multi-agent Q-learning framework for optimizing stock trading systems](https://www.researchgate.net/publication/221465347_A_Multi-agent_Q-learning_Framework_for_Optimizing_Stock_Trading_Systems) by Lee J W, Jangmin O. DEXA, 2002.
* [Multi-agent reinforcement learning for traffic light control](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=422747CB9AF552CF1C4E455220E3F96F?doi=10.1.1.32.9887&rep=rep1&type=pdf) by Wiering, Marco. ICML. 2000.










About
----
This documents will be synchronized to the following platforms
1. [CSDN-Blog: A Guide Resource for Deep Reinforcement Learning]()
2. [ZhiHu-Blog: A Guide Resource for Deep Reinforcement Learning]()





Cite
----

[1].https://github.com/brianspiering/awesome-deep-rl#talks
[2].https://github.com/jgvictores/awesome-deep-reinforcement-learning
[3].https://github.com/PaddlePaddle/PARL/blob/develop/papers/archive.md#distributed-training
[4].https://github.com/LantaoYu/MARL-Papers

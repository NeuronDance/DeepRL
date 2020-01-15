# Deep Reinforcement Learning(深度强化学习)

<p align="center">
<a href="https://travis-ci.org/onevcat/Kingfisher"><img src="https://img.shields.io/travis/onevcat/Kingfisher/master.svg"></a>
<a href="https://raw.githubusercontent.com/onevcat/Kingfisher/master/LICENSE"><img src="https://img.shields.io/cocoapods/l/Kingfisher.svg?style=flat"></a>

[](https://img.shields.io/github/issues/NeuronDance/DeepRL)

</p><br>


本仓库由“深度强化学习实验室(DeepRL-Lab)”创建，希望能够为所有DRL研究者，学习者和爱好者提供一个学习指导。


>如今机器学习发展如此迅猛，各类算法层出不群，特别是深度神经网络在计算机视觉、自然语言处理、时间序列预测等多个领域更是战果累累，可以说这波浪潮带动了很多人进入深度学习领域，也成就了其一番事业。而强化学习作为一门灵感来源于心理学中的行为主义理论的学科，其内容涉及概率论、统计学、逼近论、凸分析、计算复杂性理论、运筹学等多学科知识，难度之大，门槛之高，导致其发展速度特别缓慢。围棋作为人类的娱乐游戏中复杂度最高的一个，它横竖各有19条线，共有361个落子点，双方交替落子，状态空间高达10的171次方(注：宇宙中的原子总数是10的80次方，即使穷尽整个宇宙的物质也不能存下围棋的所有可能性）
### 1、Deep Reinforcement Learning？
时间 |   内容| 
-|-|
2015年10月 | 由Google-DeepMind公司开发的AlphaGo程序击败了人类高级选手樊麾，成为第一个无需让子即可在19路棋盘上击败围棋职业棋手的计算机围棋程序，并写进了历史，论文发表在国际顶级期刊《Science》上| 
2016年3月| 透过自我对弈数以万计盘进行练习强化，AlphaGo在一场五番棋比赛中4:1击败顶尖职业棋手李世石。|
2016年12月|Master(AlphaGo版本)开始出现于弈城围棋网和腾讯野狐围棋网，取得60连胜的成绩，以其空前的实力轰动了围棋界。|
-|DeepMind 如约公布了他们最新版AlphaGo论文(Nature)，介绍了迄今最强最新的版本AlphaGo Zero，使用纯强化学习，将价值网络和策略网络整合为一个架构，3天训练后就以100比0击败了上一版本的AlphaGo。AlphaGo已经退休，但技术永存。DeepMind已经完成围棋上的概念证明，接下来就是用强化学习创造改变世界的价值。|

围棋被攻克证明了强化学习发展的威力，作为AlphoGo的带头人，强化学习界的大神，David Sliver提出人工智能的终极目标是：

**AI = DL(Deep Learning) + RL(Reinforcement Learning) == DRL(Deep Reinforcement Learning)**


---

### 2、Application？
在深度学习已经取得了很大的进步的基础上，深度强化学习真正的发展归功于神经网络、深度学习以及计算力的提升，David就是使用了神经网络逼近值函数后提出深度强化学习（Deep Reinforcement Learning，DRL），并证明了确定性策略等。纵观近四年的ICML，NPIS等顶级会议论文，强化学习的理论进步，应用领域逐渐爆发式增广，目前已经在如下领域有了广泛使用:
>
+ 自动驾驶：自动驾驶载具（self-driving vehicle）
+ 控制论(离散和连续大动作空间): 玩具直升机、Gymm_cotrol物理部件控制、机器人行走、机械臂控制。
+ 游戏：Go, Atari 2600(DeepMind论文详解)等
+ 自然语言处理：机器翻译, 文本序列预测，问答系统，人机对话
+ 超参数学习：神经网络参数自动设计
+ 推荐系统：阿里巴巴黄皮书（商品推荐），广告投放。
+ 智能电网：电网负荷调试，调度等
+ 通信网络：动态路由, 流量分配等
+ 财务与财经系统分析与管理
+ 智能医疗
+ 智能交通网络及网络流
+ 物理化学实验：定量实验，核素碰撞，粒子束流调试等
+ 程序学习和网络安全：网络攻防等

---

### 3、一流研究机构有哪些?
机构名| Logo|官网|简介|
-|-|-|-|
DeepMind|![](assets/markdown-img-paste-20190222165835138.png)|[Access](https://deepmind.com/)|DeepMind是一家英国的人工智能公司。公司创建于2010年，最初名称是DeepMind科技（DeepMind Technologies Limited），在2014年被谷歌收购。|
OpenAI|![](assets/markdown-img-paste-20190222165707224.png)|[Access](https://openai.com/)|OpenAI是一个非营利性人工智能（AI）研究组织，旨在促进和发展友好的人工智能，使人类整体受益。这家总部位于旧金山的组织成立于2015年底，旨在通过向公众开放其专利和研究，与其他机构和研究人员“自由合作”。创始人（尤其是伊隆马斯克和萨姆奥特曼）的部分动机是出于对通用人工智能风险的担忧。|
UC Berkeley||[Access1](https://bair.berkeley.edu)<br>[Access2](http://hart.berkeley.edu/)||
...||||



### 4、业界大佬有哪些？
![](assets/markdown-img-paste-20190222191015730.png)
顺序从左往右：
>
Name|Company| Homepage|about|
-|-|-|-|
**Richard Sutton**|Deepmind|[page](http://incompleteideas.net/)|强化学习的祖师爷，著有《Reinforcement Learning: An Introduction》|
**David Sliver**|DeepMind|[page](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Home.html),[Google学术](https://scholar.google.com/citations?user=-8DNE4UAAAAJ&hl=zh-CN)|AlphaGo、AlphaStar掌门人，UCL公开课主讲人,他工作重点是将强化学习与深度学习相结合，包括一个学习直接从像素中学习Atari游戏的程序。领导AlphaGo项目，最终推出了第一个在Go全尺寸游戏中击败顶级职业玩家的计划。 AlphaGo随后获得了荣誉9丹专业认证;并因创新而获得戛纳电影节奖。然后他领导了AlphaZero的开发，它使用相同的AI来学习玩从头开始（仅通过自己玩而不是从人类游戏中学习），然后学习以相同的方式下棋和将棋，比任何其他计算机更高的水平方案|
**Oriol Vinyals**|DeepMind||AlphaStar主要负责人
**Pieter Abbeel**|UC Berkeley| [page](http://people.eecs.berkeley.edu/~pabbeel/),[Google学术](https://scholar.google.com/citations?user=vtwH6GkAAAAJ&hl=zh-CN)|机器人和强化学习专家 加州大学伯克利分校教授，EECS，BAIR，CHAI（2008-）,伯克利机器人学习实验室主任，伯克利人工智能研究（BAIR）实验室联合主任,联合创始人，总裁兼首席科学家covariant.ai（2017-）,研究科学家（2016-2017），顾问（2018-）OpenAI,联合创始人Gradescope（2014-2018：TurnItIn收购）|


### 5、如何学习?
x内容|学习方法与资料|
-|-|
补充数学基础(高数、线代、概率论)|[Access](https://github.com/NeuronDance/DeepRL/tree/master/AI-Basic-Resource)|
基础与课程学习|[Access](https://github.com/NeuronDance/DeepRL/tree/master/DRL-Course)<br>
强化学习竞赛|[Access](https://github.com/NeuronDance/DeepRL/tree/master/DRL-Competition)<br>
开源框架学习|[Access](https://github.com/NeuronDance/DeepRL/tree/master/DRL-OpenSource)






### 6、关于深度强化学习实验室
|发起人|J.Q.Wang||
-|-|-|-|-|
成员|包含教授、讲师、博士、硕士、本科、|**学术界**:清华、北大、山大、浙大、北航、东南、南大、大工、天大、中科大、北理工、国防科大、牛津大学、帝国理工、CMU、南洋理工、柏林工业、西悉尼大学、埃默里大学等<br>**工业界**:腾讯、阿里巴巴、网易、头条、华为、快手等|
愿景|DeepRL|[1]. 提供最全面的深度强化学习书籍、资料、综述等学习资源。<br>[2]. 阐述深度强化学习的基本原理、前沿算法、场景应用、竞赛分析、论文分享等专业知识。<br>[3]. 分享最前沿的业界动态和行业发展趋势。<br>[4]. 成为所有深度强化学习领域的研究者与爱好者交流平台。




### @致谢
欢迎每一位伙伴积极为项目贡献微薄之力，共同点亮星星之火。<br>


**贡献者列表(排名不分先后)**：<br>

---
@[taoyafan](https://github.com/taoyafan),@[BluesChang](https://github.com/BluesChang)，@[Wangergou123](https://github.com/Wangergou123),@[TianLin0509](https://github.com/TianLin0509)，@[zanghyu](https://github.com/zanghyu),@[hijkzzz](https://github.com/hijkzzz),@[tengshiquan](https://github.com/tengshiquan)

---

#### @联系方式
Title||
-|-|-|
微信群聊|加微信助手：NeuronDance(进交流群)|
联系邮箱|wjq_z@qq.com|
CSDN博客|[深度强化学习(DRL)探索](https://blog.csdn.net/gsww404)<br>|
知乎专栏|[DeepRL基础探索](https://zhuanlan.zhihu.com/deeprl)/[DeepRL前沿论文解读](https://zhuanlan.zhihu.com/drl-paper)
微信公众号|如下图
![](assets/markdown-img-paste-20190222165438977.png)

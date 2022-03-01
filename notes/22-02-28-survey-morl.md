# A Brief Survey for the Project
[TOC]

## 1. Paper List
### 1.1. Multi-Objective / Constrained RL
**Markov Decision Processes with Multiple Objectives**\
Krishnendu ChatterjeeRupak MajumdarThomas A. Henzinger\
*STACS, 2006*
[[paper](https://link.springer.com/chapter/10.1007/11672142_26)]

**A Survey of Multi-Objective Sequential Decision-Making**\
Diederik Marijn Roijers, Peter Vamplew, Shimon Whiteson, Richard Dazeley\
*Journal Of Artificial Intelligence Research, 2013*
[[paper](https://arxiv.org/abs/1402.0590)]

**A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation**\
Runzhe Yang, Xingyuan Sun, Karthik Narasimhan\
*NeurIPS, 2019*\
[[paper](https://arxiv.org/abs/1908.08342)] [[code](https://github.com/RunzheYang/MORL)]

**Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control**\
Jie Xu, Yunsheng Tian, Pingchuan Ma, Daniela Rus, Shinjiro Sueda, Wojciech Matusik\
*ICML, 2020*\
[[paper](https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf)] [[website](http://pgmorl.csail.mit.edu/)] [[supp](https://people.csail.mit.edu/jiex/papers/PGMORL/supp.pdf)]  [[video](https://people.csail.mit.edu/jiex/papers/PGMORL/video.mp4)]  [[code](https://github.com/mit-gfx/PGMORL)]  [[talk](https://people.csail.mit.edu/jiex/papers/PGMORL/talk.mp4)]  [[bib](http://pgmorl.csail.mit.edu/)]

**A Constrained Multi-Objective Reinforcement Learning Framework**\
DeepMind\
*Proceedings of the 5th Conference on Robot Learning, 2022*\
[[paper](https://proceedings.mlr.press/v164/huang22a.html)] [[website](https://sites.google.com/view/cmorl/home?authuser=0)]

**Safety Gym**
OpenAI\
*Blog, 2019*
[[link](https://openai.com/blog/safety-gym/)]

**Constrained versus Multi-Objective Reinforcement Learning: A Fundamental Difference?**\
Markus Peschl\
*Blog, 2021*\
[[link](https://mlpeschl.com/post/multi_objective_rl/)]



### 1.2. Multi-Goal / Goal-Conditioned / Meta RL
**InfoBot: Transfer and Exploration via the Information Bottleneck**\
Anirudh Goyal, Riashat Islam, DJ Strouse, Zafarali Ahmed, Hugo Larochelle, Matthew Botvinick, Yoshua Bengio, Sergey Levine\
*ICLR, 2019*\
[[paper](https://openreview.net/forum?id=rJg8yhAqKm)] [[code](https://github.com/maximecb/gym-minigrid)]

### 1.3. Multi-Task RL
**A Survey of Multi-Task Deep Reinforcement Learning**\
Nelson Vithayathil Varghese, Qusay H. Mahmoud\
*Electronics, 2020*\
[[paper](https://www.mdpi.com/2079-9292/9/9/1363)]


### 1.4. Information Theory in RL
**Reinforcement Learning, Bit by Bit**\
Xiuyuan Lu, Benjamin Van Roy, Vikranth Dwaracherla, Morteza Ibrahimi, Ian Osband, Zheng Wen\
*ArXiv, 2022*\
[[paper](https://arxiv.org/abs/2103.04047)]
> - The high level idea comes from Posterior Sampling, that we sample the environment from a maintained posterior at the beginning of each episode, and then take the action which minimizing the information ratio.
> - The information ratio is defined by: $$\Gamma_{\tau, t}=\frac{\mathbb{E}\left[V_{*}\left(H_{t}\right)-Q_{*}\left(H_{t}, A_{t}\right)\right]^{2}}{\left(\mathbb{I}\left(\chi ; \mathcal{E} \mid P_{t}\right)-\mathbb{I}\left(\chi ; \mathcal{E} \mid P_{t+\tau}\right)\right) / \tau}.$$
> - This can be view as a very classical exploration-exploitation tradeoff. The numerator represents the immediate shortfall, while the nominator represents the information gain about the environment.
> - :baby_chick:: A potential improvement might be incorporating information bottleneck in the denominator part. That is, we may not need to learn that much from the environment: we only want learn those to be relevant to our objective, and discard those irrelevant.

**InfoBot: Transfer and Exploration via the Information Bottleneck**\
Anirudh Goyal, Riashat Islam, DJ Strouse, Zafarali Ahmed, Hugo Larochelle, Matthew Botvinick, Yoshua Bengio, Sergey Levine\
*ICLR, 2019*\
[[paper](https://openreview.net/forum?id=rJg8yhAqKm)] [[code](https://github.com/maximecb/gym-minigrid)]
> - The idea is simply to constrain the dependence on a certain goal, so that the agent can learn a *default behavior*.
> - This done by introducing $- \beta I(A ; G \mid S)$ or equivalently $- \beta D_{\mathrm{KL}}\left[\pi_{\theta}(A \mid S, G) \mid \pi_{0}(A \mid S)\right]$ in the reward function, that is: $$
\begin{aligned}
J(\theta) & \equiv \mathbb{E}_{\pi_{\theta}}[r]-\beta I(A ; G \mid S) \\
&=\mathbb{E}_{\pi_{\theta}}\left[r-\beta D_{\mathrm{KL}}\left[\pi_{\theta}(A \mid S, G) \mid \pi_{0}(A \mid S)\right]\right].
\end{aligned}
$$

**Dynamics Generalization via Information Bottleneck in Deep Reinforcement Learning**\
Xingyu Lu, Kimin Lee, Pieter Abbeel, Stas Tiomkin\
[[paper](https://arxiv.org/abs/2103.04047)]

**Dynamic Bottleneck for Robust Self-Supervised Exploration**\
Chenjia Bai, Lingxiao Wang, Lei Han, Animesh Garg, Jianye HAO, Peng Liu, Zhaoran Wang\
*NeurIPS, 2021*\
[[paper](https://openreview.net/forum?id=-t6TeG3A6Do)] [[code](https://github.com/Baichenjia/DB)]
> - The high-level idea is to first generate a dynamics-relevant representation $Z_{t}$, then impose a bottleneck on dynamics, in order to through away the dynamics-irrelevant information.
> - That is, $\min I([S_t, A_t]; Z_t)$, and $\max I(Z_t; S_{t+1})$.

## 2. A Unified View
Below are cited from this [post](https://mlpeschl.com/post/multi_objective_rl/) and this [paper](https://arxiv.org/pdf/1908.08342.pdf).

### 2.1. MDP
$$
\max _{\pi} J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{T} \gamma^{t} r\left(s_{t}, a_{t}\right)\right]
$$
### 2.2. Constrained MDP (CMDP)
$$
\max _{\pi} J(\pi) \\ \text { s.t. }  J_{\mathbf{c}}(\pi) \leq \mathbf{d}
$$

### 2.3. Multi-Objective MDP (MOMDP)
According to a preference vector $\omega \in \Omega$, we consider $f_{\boldsymbol{\omega}}(\mathbf{r}(\mathbf{s}, \mathbf{a}))=\boldsymbol{\omega}^{T} \mathbf{r}(\boldsymbol{s}, \boldsymbol{a})$:
$$
\mathcal{F}:=\left\{J_{\mathbf{r}}(\pi) \mid \pi \in \Pi \wedge \nexists \pi^{\prime} \neq \pi: J_{\mathbf{r}}\left(\pi^{\prime}\right) \geq J_{\mathbf{r}}(\pi)\right\}.
$$

# A Brief Survey for the Project
[TOC]

## Paper List
### Multi-Objective RL
**Markov Decision Processes with Multiple Objectives**\
Krishnendu ChatterjeeRupak MajumdarThomas A. Henzinger\
*STACS, 2006*
[[paper](https://link.springer.com/chapter/10.1007/11672142_26)]

**A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation**\
Runzhe Yang, Xingyuan Sun, Karthik Narasimhan\
*NeurIPS, 2019*\
[[paper](https://arxiv.org/abs/1908.08342)] [[code](https://github.com/RunzheYang/MORL)]

**Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control**\
Jie Xu, Yunsheng Tian, Pingchuan Ma, Daniela Rus, Shinjiro Sueda, Wojciech Matusik\
*ICML, 2020*\
[[paper](https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf)]  [[supp](https://people.csail.mit.edu/jiex/papers/PGMORL/supp.pdf)]  [[video](https://people.csail.mit.edu/jiex/papers/PGMORL/video.mp4)]  [code](https://github.com/mit-gfx/PGMORL)]  [[talk](https://people.csail.mit.edu/jiex/papers/PGMORL/talk.mp4)]  [[bib](http://pgmorl.csail.mit.edu/)]

**A Constrained Multi-Objective Reinforcement Learning Framework**\
DeepMind\
*Proceedings of the 5th Conference on Robot Learning, 2022*\
[[paper](https://proceedings.mlr.press/v164/huang22a.html)]

### Multi-Goal or Goal-Conditioned or Meta RL
**InfoBot: Transfer and Exploration via the Information Bottleneck**\
Anirudh Goyal, Riashat Islam, DJ Strouse, Zafarali Ahmed, Hugo Larochelle, Matthew Botvinick, Yoshua Bengio, Sergey Levine\
*ICLR, 2019*\
[[paper](https://openreview.net/forum?id=rJg8yhAqKm)] [[code](https://github.com/maximecb/gym-minigrid)]
> The idea is simply to constrain the dependence on a certain goal, by introducing $- \beta I(A ; G \mid S)$ or equivalently $- \beta D_{\mathrm{KL}}\left[\pi_{\theta}(A \mid S, G) \mid \pi_{0}(A \mid S)\right]$ in the reward function.

### Information Bottleneck in RL
**Dynamic Bottleneck for Robust Self-Supervised Exploration**\
Chenjia Bai, Lingxiao Wang, Lei Han, Animesh Garg, Jianye HAO, Peng Liu, Zhaoran Wang\
*NeurIPS, 2021*\
[[paper](https://openreview.net/forum?id=-t6TeG3A6Do)] [[code](https://github.com/Baichenjia/DB)]
> - The high-level idea is to first generate a dynamics-relevant representation $Z_{t}$, then impose a bottleneck on dynamics, in order to through away the dynamics-irrelevant information.
> - That is, $\min I([S_t, A_t]; Z_t)$, and $\max I(Z_t; S_{t+1})$.

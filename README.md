MATD3 — Paper Implementation 
A from-scratch implementation of the Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) algorithm from Wen &
Kok (2020), built as a learning exercise to understand the mechanics of TD3 extended to the multi-agent setting. 
 Purpose 
This repository is a direct implementation of the MATD3 paper — written to understand the algorithm from the inside out rather
than use an existing library. It preceded and directly motivated the more advanced sparse-comm-swarm-rl project, which extends
these ideas with bandwidth-aware gated execution for edge deployment. 
 What MATD3 Does 
MATD3 extends TD3 into the multi-agent cooperative setting using Centralised Training with Decentralised Execution (CTDE). 
The three TD3 mechanisms implemented: 
1. Twin Critics with Clipped Q-targets Two independent critic networks estimate Q-values. The minimum of the two is used for
target computation — directly countering the overestimation bias inherited by MADDPG from DDPG. 
2. Delayed Policy Updates The actor updates every 2 critic steps. Prevents the policy from chasing a still-noisy value function early in
training. 
3. Target Policy Smoothing Small clipped noise added to target actions before computing target Q-values. Smooths the value
landscape and prevents critic overfitting to narrow action regions. 
 Environment 
OpenAI Gym multi-agent environments with continuous action spaces. 
 Project Structure 
MADDPG/
├── maddpg.py # MATD3 training logic
├── networks.py # Actor and Critic architectures
├── buffer.py # Experience replay buffer
└── train.py # Training entry point

 Quickstart 
pip install torch numpy gym
 python train.py

 Relation To Other Work 
This implementation identified two limitations worth addressing: 
No exploration noise wired into the training loop
Standard replay buffer without prioritisation 
Both were addressed in the follow-up project: sparse-comm-swarm-rl 
Reference 
Wen & Kok, Multi-Agent Twin Delayed Deep Deterministic Policy Gradient, AAMAS 2020. arxiv.org/abs/2004.0967

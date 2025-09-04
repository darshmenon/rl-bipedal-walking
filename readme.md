# 🦿 RL Bipedal Walking

**`rl-bipedal-walking`** is a reinforcement learning project where an agent learns to walk using a simplified **2D bipedal humanoid model**.  
The project builds on the **BipedalWalker-v3** environment from [Gymnasium](https://gymnasium.farama.org/), where the agent must discover stable walking gaits through trial and error.

---

## 🚀 Features
- Train a bipedal walker with **RL algorithms** (PPO, SAC, DDPG).  
- Explore **basic locomotion** → walking forward on flat terrain.  
- Extend to **challenging terrains** → uneven ground, slopes, obstacles.  
- Save and load trained models for evaluation.  
- Visualize training progress with reward plots and episode rollouts.  

---

## 📦 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/darshmenon/rl-bipedal-walking.git
cd rl-bipedal-walking
pip install -r requirements.txt
````

**requirements.txt** (minimal):

```
gymnasium[box2d]
stable-baselines3
torch
numpy
matplotlib
```

---

## 🏃 Training

Train a PPO agent on the **BipedalWalker-v3** environment:

```bash
python train.py
```

This will:

* Initialize the environment
* Train the agent for `1e6` timesteps
* Save the trained model in the `models/` directory

---

## 🎮 Evaluation

Render a trained agent:

```bash
python evaluate.py
```

---

## 📊 Results

(Coming soon – add plots/gifs of your walker here 🚶)

---

## 🔮 Extensions

* 🌄 Uneven terrain / slopes
* 🪜 Obstacle navigation
* ⚡ Energy-efficient gait learning
* 👫 Multi-agent bipedal walkers (competition or cooperation)

---

## 📌 References

* [Gymnasium BipedalWalker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

---

👨‍💻 Author: [Darsh Menon](https://github.com/darshmenon)

```


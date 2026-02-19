# Sales Negotiation DQN

Reinforcement learning project that trains a DQN (Deep Q-Network) agent to handle customer objections in a sales negotiation environment.

## Overview

The agent learns to optimize sales outcomes by choosing among three actions at each step:

- **Persuade** (0): Attempt to resolve the current objection through persuasion
- **Incentive** (1): Offer a cost-based incentive (A, B, or C) to overcome the objection
- **Closing** (2): End the call (early stopping)

Customer objections are categorized as A (quality), B (price), C (logistics), or D (other). The environment models latent variables (customer patience, determination) that affect conversion probability.

## Project Structure

```
├── gym_env.py           # SalesNegotiationEnv - Gymnasium environment
├── eval.py              # Evaluation utilities (FlattenObservationWrapper, evaluate_strategy)
├── DQN_training.ipynb   # DQN training pipeline using d3rlpy
├── generate_convo.ipynb # Conversation generation using Google GenAI
├── requirements.txt
└── d3rlpy_logs/        # Saved models and training logs
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For conversation generation, create a `.env` file with:
   ```
   GOOGLE_API_KEY=your_key
   ```

## Usage

- **Training**: Run cells in `DQN_training.ipynb` to train a DQN agent on `SalesNegotiationEnv`. Models are saved under `d3rlpy_logs/`.

- **Evaluation**: Use `eval.py` to evaluate a trained model:
  ```python
  from eval import evaluate_strategy, FlattenObservationWrapper
  from gym_env import SalesNegotiationEnv

  base_env = SalesNegotiationEnv(max_round=30)
  env = FlattenObservationWrapper(base_env)
  rewards, rounds = evaluate_strategy(model, env, n_episodes=100)
  ```

- **Conversation generation**: Use `generate_convo.ipynb` to generate customer profiles and dialogue data with Google GenAI.

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `labor_cost` | 0.2 | Cost per round |
| `incentive_A/B/C_cost` | 2, 5, 7 | Cost of each incentive type |
| `profit` | 10 | Revenue on successful sale |
| `exit_penalty` | -2 | Penalty for early exit |
| `max_round` | 30 | Maximum conversation rounds |
| `exit_start_round` | 5 | Earliest round for customer churn |

## Dependencies

Key packages: `gymnasium`, `d3rlpy`, `torch`, `pandas`, `numpy`. See `requirements.txt` for full list.

import random
import json
from pathlib import Path
import matplotlib.pyplot as plt

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
EPISODES = 100

# Simple PPE compliance environment
# States:
# 0 = compliant
# 1 = missing hardhat
# 2 = missing vest
# 3 = missing mask
#
# Actions:
# 0 = do nothing
# 1 = raise alert

NUM_STATES = 4
NUM_ACTIONS = 2

def reward(state, action):
    if state == 0 and action == 0:
        return 1
    if state != 0 and action == 1:
        return 1
    return -1

def next_state():
    return random.randint(0, NUM_STATES - 1)

def choose_action(q_table, state):
    if random.random() < EPSILON:
        return random.randint(0, NUM_ACTIONS - 1)
    return max(range(NUM_ACTIONS), key=lambda a: q_table[state][a])

def train_agent():
    q_table = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    episode_rewards = []

    for _ in range(EPISODES):
        state = next_state()
        total_reward = 0

        for _ in range(20):
            action = choose_action(q_table, state)
            r = reward(state, action)
            new_state = next_state()

            best_next = max(q_table[new_state])
            old_q = q_table[state][action]

            q_table[state][action] = old_q + ALPHA * (
                r + GAMMA * best_next - old_q
            )

            state = new_state
            total_reward += r

        episode_rewards.append(total_reward)

    return q_table, episode_rewards

def save_learning_curve(rewards, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(rewards)
    plt.title("Q-Learning Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    q_table, rewards = train_agent()

    output_dir = Path("models/rl")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_learning_curve(rewards, output_dir / "learning_curve.png")

    with open(output_dir / "q_table.json", "w", encoding="utf-8") as f:
        json.dump(q_table, f, indent=2)

    print("RL prototype complete.")
    print(f"Saved Q-table to: {output_dir / 'q_table.json'}")
    print(f"Saved learning curve to: {output_dir / 'learning_curve.png'}")

if __name__ == "__main__":
    main()
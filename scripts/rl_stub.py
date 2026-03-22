def reward_function(correct_detection):
    return 1 if correct_detection else -1


def simulate_learning_curve():
    sample_cases = [True, False, True, True, False, True]
    rewards = [reward_function(case) for case in sample_cases]

    cumulative_rewards = []
    total = 0
    for reward in rewards:
        total += reward
        cumulative_rewards.append(total)

    return rewards, cumulative_rewards


if __name__ == "__main__":
    rewards, cumulative = simulate_learning_curve()
    print("Rewards:", rewards)
    print("Cumulative rewards:", cumulative)
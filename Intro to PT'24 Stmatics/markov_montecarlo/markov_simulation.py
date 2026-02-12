# markov_simulation.py

import numpy as np
import random
import matplotlib.pyplot as plt

# --- Define Markov Chain States and Transition Matrix ---
states = ['Sunny', 'Rainy', 'Cloudy']
transition_matrix = [
    [0.7, 0.2, 0.1],  # Sunny -> [Sunny, Rainy, Cloudy]
    [0.3, 0.4, 0.3],  # Rainy -> ...
    [0.2, 0.3, 0.5]   # Cloudy -> ...
]

# --- Simulate Markov Chain ---
def run_markov_chain(start_state='Sunny', steps=1000):
    current_state = states.index(start_state)
    state_counts = [0] * len(states)

    for _ in range(steps):
        state_counts[current_state] += 1
        current_state = np.random.choice([0, 1, 2], p=transition_matrix[current_state])

    proportions = [count / steps for count in state_counts]

    print("Estimated long-term probabilities after", steps, "steps:")
    for state, prob in zip(states, proportions):
        print(f"{state}: {prob:.3f}")

    return proportions

# --- Plot Results ---
def plot_distribution(proportions):
    plt.bar(states, proportions)
    plt.title("Estimated Steady-State Distribution (Markov Chain)")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.show()

# --- Main ---
if __name__ == "__main__":
    proportions = run_markov_chain(start_state='Sunny', steps=10000)
    plot_distribution(proportions)

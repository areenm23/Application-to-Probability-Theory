# montecarlo_simulation.py

import random
import matplotlib.pyplot as plt

# --- Estimate Pi Using Monte Carlo ---
def estimate_pi(num_samples=10000):
    inside_circle = 0
    x_vals = []
    y_vals = []
    colors = []

    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = x**2 + y**2
        if distance <= 1:
            inside_circle += 1
            colors.append('blue')  # Inside
        else:
            colors.append('red')   # Outside
        x_vals.append(x)
        y_vals.append(y)

    pi_estimate = 4 * inside_circle / num_samples
    print(f"Estimated value of π after {num_samples} samples: {pi_estimate:.5f}")
    return pi_estimate, x_vals, y_vals, colors

# --- Plot Results ---
def plot_points(x_vals, y_vals, colors):
    plt.figure(figsize=(6,6))
    plt.scatter(x_vals, y_vals, c=colors, s=1)
    plt.title("Monte Carlo Estimation of π")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# --- Main ---
if __name__ == "__main__":
    pi, x_vals, y_vals, colors = estimate_pi(num_samples=10000)
    plot_points(x_vals, y_vals, colors)

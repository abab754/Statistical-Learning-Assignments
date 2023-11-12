import numpy as np
import matplotlib.pyplot as plt

# Define the matrices A1, A2, A3, A4
A1 = np.array([[0, 0], [0, 0.16]])
A2 = np.array([[0.2, -0.26], [0.23, 0.22]])
A3 = np.array([[-0.15, 0.28], [0.26, 0.24]])
A4 = np.array([[0.85, 0.04], [-0.04, 0.85]])

# Define the vectors b1, b2, b3, b4
b1 = np.array([0, 0])
b2 = np.array([0, 0.16])
b3 = np.array([0, 0.44])
b4 = np.array([0, 1.6])

# Set the number of samples
n = 100000

# Initialize the matrix to store X_t values
X = np.zeros((n, 2))
# Set initial value of X
X[0] = [0, 0]

# Define the probabilities for theta
prob_theta = [0.01, 0.07, 0.07, 0.85]

# Run the Markov Chain for n iterations
np.random.seed(123)  # for reproducibility
for t in range(1, n):
    # Sample theta_t based on the given probabilities
    theta_t = np.random.choice([1, 2, 3, 4], p=prob_theta)

    # Compute X_t+1 based on the sampled theta_t
    if theta_t == 1:
        X[t] = np.dot(A1, X[t - 1]) + b1
    elif theta_t == 2:
        X[t] = np.dot(A2, X[t - 1]) + b2
    elif theta_t == 3:
        X[t] = np.dot(A3, X[t - 1]) + b3
    elif theta_t == 4:
        X[t] = np.dot(A4, X[t - 1]) + b4

# Plot the generated Markov chain
plt.scatter(X[:, 0], X[:, 1], s=0.05)
plt.axis('equal')
plt.show()
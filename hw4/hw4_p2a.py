import pandas as pd
import numpy as np

def objectiveValue(X, Y, beta, lambda_):
    xb = np.dot(X, beta)
    obj = np.sum(-Y * xb + np.log(1 + np.exp(xb)) + lambda_ * np.linalg.norm(beta)**2)
    return obj

def logisticReg_gradientDescent(X, Y, lambda_):
    n, p = X.shape
    beta_prev = np.zeros(p)
    maxiter = 100000
    stepsize = 0.01 / n

    for t in range(1, maxiter + 1):
        xb = np.dot(X, beta_prev)
        gradient = -np.dot(X.T, (Y - np.exp(xb) / (1 + np.exp(xb)))) + 2 * lambda_ * beta_prev
        beta_next = beta_prev - stepsize * gradient

        if np.linalg.norm(gradient, ord=2) < 1e-4:
            print(f"Converged at iteration: {t}")
            break
        else:
            beta_prev = beta_next

        cur_obj = objectiveValue(X, Y, beta_next, lambda_)

        if t % 5000 == 0:
            print(f"Current iter: {t}, Current objective value: {cur_obj}")

        if t == maxiter:
            print("Reached maxiter; did not converge.")

    print(f"Final gradient norm: {np.linalg.norm(gradient, ord=2)}")
    return beta_next

# Load the data
votes = pd.read_csv("votes.csv")

# Filter and preprocess data
votes = votes[votes['state_abbr'].isin(["TX", "NY", "PA", "CA"])]
votes['texas'] = (votes['state_abbr'] == "TX")
n = len(votes)

# Set seed and shuffle data
np.random.seed(3)
votes = votes.sample(frac=1).reset_index(drop=True)

# Define features
features = ["white", "hispanic", "poverty",
            "bachelor", "highschool", "age18under",
            "female", "landarea", "income", "asian",
            "population2014", "density", "household_size",
            "veteran", "age65plus", "black",
            "immigrant"]

Y = votes['texas']
X = np.column_stack([np.ones(n), votes[features]])
p = X.shape[1]

# Standardize features
for j in range(1, p):
    X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])

n_learn = 200
n_test = 200

Y_test = Y[:n_test]
Y_learn = Y[n_test:n_test + n_learn]

X_test = X[:n_test, :]
X_learn = X[n_test:n_test + n_learn, :]

# Using logistic regression function
lambda_ = 1
beta_lambda = logisticReg_gradientDescent(X_learn, Y_learn, lambda_)

# Making predictions on test data and computing the misclassification error
Ypred = (np.dot(X_test, beta_lambda) > 0)
print("Test misclassification error:")
print(np.mean(Y_test != Ypred))
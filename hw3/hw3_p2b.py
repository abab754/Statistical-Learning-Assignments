import pandas as pd
import numpy as np

# Read the CSV
movies = pd.read_csv("../movies.csv")

# Shuffle the rows
np.random.seed(4)
movies = movies.sample(frac=1).reset_index(drop=True)

# Filter out movies with budget in the bottom 10 percentile
# and movies with missing runtime
min_budget_threshold = movies['budget'].quantile(0.1)
movies = movies[movies['budget'] > min_budget_threshold]
movies = movies.dropna(subset=['runtime'])

# Create three new features
movies['age'] = (pd.to_datetime("10/12/2021") - pd.to_datetime(movies['release_date'])).dt.days
movies['title_len'] = movies['title'].apply(len)
movies['log_budget'] = np.log(movies['budget'])

# Selecting features
features = ["Drama", "Action", "Comedy", "Horror", "log_budget", "runtime", "age", "title_len"]
X = movies[features].copy()
Y = movies['vote_average']

# Adding a constant column to X
X['constant'] = 1

# Split data for testing and learning
n_test = 500
n_learn = 1500

Y_test = Y.iloc[:n_test]
Y_learn = Y.iloc[n_test:n_test + n_learn]
X_test = X.iloc[:n_test]
X_learn = X.iloc[n_test:n_test + n_learn]

K = 10  # Number of folds, we require that n_learn is divisible by K
kf = range(K)

validerr = np.zeros((K, 2))

for k in kf:

    n_valid = n_learn // K

    valid_ix = list(range(k * n_valid, (k + 1) * n_valid))
    train_ix = list(set(range(n_learn)) - set(valid_ix))

    X_valid = X_learn.iloc[valid_ix].values
    X_train = X_learn.iloc[train_ix].values
    Y_valid = Y_learn.iloc[valid_ix].values
    Y_train = Y_learn.iloc[train_ix].values

    # Fit two separated linear regression models
    ixs = X_train[:, 0] == 1
    X_train1 = X_train[ixs, 1:]
    X_train2 = X_train[~ixs, 1:]
    Y_train1 = Y_train[ixs]
    Y_train2 = Y_train[~ixs]

    beta1 = np.linalg.lstsq(X_train1, Y_train1, rcond=None)[0]
    beta2 = np.linalg.lstsq(X_train2, Y_train2, rcond=None)[0]

    ixs = X_valid[:, 0] == 1
    X_valid1 = X_valid[ixs, 1:]
    X_valid2 = X_valid[~ixs, 1:]
    Y_validpred = np.zeros(n_valid)
    Y_validpred[ixs] = X_valid1 @ beta1
    Y_validpred[~ixs] = X_valid2 @ beta2
    validerr[k, 0] = np.mean((Y_validpred - Y_valid) ** 2)

    # Fit linear regression on all data using "Drama" as a 0/1 feature
    beta_all = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
    Y_validpred = X_valid @ beta_all
    validerr[k, 1] = np.mean((Y_validpred - Y_valid) ** 2)

# Calculate mean validation error
mean_validerr = validerr.mean(axis=0)
print(f"Separated linreg error: {mean_validerr[0]:.3f}")
print(f"Single linreg error: {mean_validerr[1]:.3f}")

# Create final prediction using the best model on X_test, Y_test
if mean_validerr[0] < mean_validerr[1]:
    ixs = X_test.iloc[:, 0].values == 1
    X_test1 = X_test[ixs].iloc[:, 1:]
    X_test2 = X_test[~ixs].iloc[:, 1:]
    beta1 = np.linalg.lstsq(X_train1, Y_train1, rcond=None)[0]
    beta2 = np.linalg.lstsq(X_train2, Y_train2, rcond=None)[0]
    Y_pred = np.zeros(n_test)
    Y_pred[ixs] = X_test1 @ beta1
    Y_pred[~ixs] = X_test2 @ beta2
else:
    X_test_combined = X_test.values
    beta_all = np.linalg.lstsq(X_learn.values, Y_learn.values, rcond=None)[0]
    Y_pred = X_test_combined @ beta_all

# Evaluate test error
test_error = np.sqrt(np.mean((Y_test - Y_pred) ** 2))
baseline_error = np.sqrt(np.mean((Y_test - np.mean(Y_learn)) ** 2))

print("Test Error vs Baseline Error")
print(test_error, baseline_error)

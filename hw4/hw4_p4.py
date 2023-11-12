import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_absolute_error

# Set seed for reproducibility
np.random.seed(1)

# Read the CSV file
shaq = pd.read_csv("shaq.csv")

# Shuffle the DataFrame
shaq = shaq.sample(frac=1)

# Part (a)
# Compute the following:

# 1. Average accuracy of Shaq's free throw
average_accuracy = shaq['shot_made'].mean()

print(f"Average accuracy of Shaq's free throw: {average_accuracy:.4f}")

# 2. Average accuracy of Shaq's free throw during a home game
home_game_accuracy = shaq[shaq['home_game'] == 1]['shot_made'].mean()

print(f"Average accuracy of Shaq's free throw during a home game: {home_game_accuracy:.4f}")

# 3. Average accuracy of Shaq's free throw when the free throw is the first of the two free throws.
first_shot_accuracy = shaq[shaq['first_shot'] == 1]['shot_made'].mean()
print(f"Average accuracy of Shaq's free throw when it's the first shot: {first_shot_accuracy:.4f}")

# Part (b)
features = ["first_shot", "missed_first", "home_game", "cur_score",
            "opp_score", "cur_time", "score_ratio",
            "made_first", "losing"]

# Note that X has no intercept
X = shaq[features].to_numpy()
Y = shaq["shot_made"].to_numpy()

n_learn = 800
n_test = 800

X_learn = X[:n_learn, :]
Y_learn = Y[:n_learn]

X_test = X[n_learn:n_learn + n_test, :]
Y_test = Y[n_learn:n_learn + n_test]

# automatic cross-validation from glmnet package
cv_result = LogisticRegressionCV(cv=5, penalty='l2', solver='liblinear')
cv_result.fit(X_learn, Y_learn)

# lambda that attains the smallest cross-validation error
lambda_min = 1.0 / (2.0 * cv_result.C_[0])

# Fit logistic regression with the selected lambda
glmnet_result = LogisticRegression(penalty='l2', C=1.0 / (2.0 * lambda_min), solver='liblinear')
glmnet_result.fit(X_learn, Y_learn)

beta = glmnet_result.coef_[0]
intercept = glmnet_result.intercept_[0]

Y_pred = glmnet_result.predict(X_test)
test_err = mean_absolute_error(Y_test, Y_pred)

if np.mean(Y_learn) > 0.5:
    Y_baseline = np.ones(n_test)
else:
    Y_baseline = np.zeros(n_test)

baseline_err = mean_absolute_error(Y_test, Y_baseline)

print(f"Ridge error: {test_err:.4f}   Baseline error: {baseline_err:.4f}")


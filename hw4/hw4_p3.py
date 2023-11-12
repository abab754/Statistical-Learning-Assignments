import pandas as pd
import numpy as np
## you may want to install skilit-learn first, try pip install -U scikit-learn
from sklearn import svm

# Read the CSV file into a DataFrame
votes = pd.read_csv("votes.csv")

# Filter rows where state_abbr is in ["TX", "NY", "PA", "CA"] and create a new column 'texas'
votes = votes[votes['state_abbr'].isin(["TX", "NY", "PA", "CA"])]
votes['texas'] = votes['state_abbr'] == "TX"

# Get the number of rows
n = len(votes)

# Set seed for reproducibility
np.random.seed(2)

# Shuffle the DataFrame
votes = votes.sample(n=n, replace=False)

# Define the list of features
features = ["white", "hispanic", "poverty", "bachelor", "highschool", "age18under",
            "female", "landarea", "income", "asian", "population2014", "density",
            "household_size", "veteran", "age65plus", "black"]

# Create the target variable Y and feature matrix X
Y = votes['texas'].values
X = votes[features].values
p = X.shape[1]

# Standardize the feature matrix
for j in range(1, p):
    X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])




# Define the number of samples for learning and testing
n_learn = 200
n_test = 200

# Split Y into test and learn sets
Y_test = Y[:n_test]
Y_learn = Y[n_test:n_test + n_learn]

# Split X into test and learn sets
X_test = X[:n_test, :]
X_learn = X[n_test:n_test + n_learn, :]

# Define the number of folds for cross-validation
K = 10

# Create folds for cross-validation
folds = np.array_split(np.arange(1, n_learn + 1), K)

# Define a list of candidate values for C
candidate_Cs = 2 ** np.arange(-5, 7, 0.7)

# Initialize a matrix to store validation errors
validerr = np.zeros((K, len(candidate_Cs)))

for k in range(K):
    for j, C in enumerate(candidate_Cs):
        valid_ix = folds[k]
        train_ix = np.setdiff1d(np.arange(1, n_learn + 1), valid_ix)

        X_valid = X_learn[valid_ix - 1, :]
        X_train = X_learn[train_ix - 1, :]

        Y_valid = Y_learn[valid_ix - 1]
        Y_train = Y_learn[train_ix - 1]

        # Create and train an SVM model with linear kernel
        svm_obj = svm.SVC(C=C, kernel='linear')
        svm_obj.fit(X_train, Y_train)

        # Predict using the trained model
        Yvalidpred = svm_obj.predict(X_valid)

        # Calculate validation error
        validerr[k, j] = np.mean(Yvalidpred != Y_valid)

# Calculate mean validation errors for each C
mean_validerr = np.mean(validerr, axis=0)

# Find the C value with the minimum mean validation error
C_min = candidate_Cs[np.argmin(mean_validerr)]

# Train the final SVM model on the entire learning set with the selected C value
final_svm_obj = svm.SVC(C=C_min, kernel='linear')
final_svm_obj.fit(X_learn, Y_learn)

# Predict using the final model on the test set
Ypred = final_svm_obj.predict(X_test)

# Calculate the test error
svm_testerr =  np.mean(Ypred != Y_test)

print("SVM test error")
print(svm_testerr)

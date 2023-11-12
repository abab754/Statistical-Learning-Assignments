import numpy as np

# NAME: Abhiram Banda
# NetID: ab2256

# Instructions:
# Complete the following Python program by filling in lines labeled ___FILL IN____.
# Each ___FILL IN____ should be only one line, but you may add additional lines of code if you think it is necessary.

n_sim = 3000

my_viewsA = 60
my_viewsB = 31
my_viewsC = 41

# Calculate the total number of views
all_views = my_viewsA + my_viewsB + my_viewsC

n_impsA = 160
n_impsB = 120
n_impsC = 150

# Calculate the total number of impressions
all_imps = n_impsA + n_impsB + n_impsC

# Calculate the CTRs for each title
my_CTRs = [my_viewsA / n_impsA, my_viewsB / n_impsB, my_viewsC / n_impsC]

# Calculate the test statistic
my_deviation = max(my_CTRs) - np.mean(my_CTRs)

# Initialize an array to store the results
all_results = np.zeros(n_sim)

# Begin simulation
for cur_sim in range(n_sim):
    # Create the pool of all impressions
    pool = np.array([1] * all_views + [0] * (all_imps - all_views))
    np.random.shuffle(pool)

    # Sample for title A
    impsA = np.random.choice(pool, n_impsA, replace=False)
    viewsA = np.sum(impsA)

    # Update the pool for titles B and C
    pool_BC = pool.copy()  # Create a copy of the original pool
    pool_BC[impsA] = 0  # Set impressions for title A to 0

    # Sample for title B
    impsB = np.random.choice(pool_BC, n_impsB, replace=False)
    viewsB = np.sum(impsB)

    # Compute views for title C
    viewsC = all_views - viewsA - viewsB

    # Calculate the CTRs for this simulation
    CTRs = [viewsA / n_impsA, viewsB / n_impsB, viewsC / n_impsC]

    # Calculate the test statistic for this simulation
    deviation = max(CTRs) - np.mean(CTRs)

    # Store the result
    all_results[cur_sim] = deviation


# Compute the p-value
p_value = (all_results >= my_deviation).mean()

print("Simulated p-value:")
print(p_value)
# NAME: Abhiram Banda
# NetID: ab2256
#
# Instructions:
#
# Complete the following Python program by
# filling in lines labled __FILL IN___
#
# Each ___FILL IN____ should be at most two lines but you
# may add additional lines of code if you think it is
# necessary.
#

import numpy as np

# Initialize variables
my_wins = 4
my_ties = 2
my_loss = 1

# ___FILL IN____
n_games = my_wins + my_ties + my_loss

# ___FILL IN____
my_point = my_wins - my_loss

n_sim = 2000

all_results = np.zeros(n_sim)  # Create an array filled with zeros

for cur_sim in range(n_sim):
    outcome = np.random.choice([-1, 0, 1], n_games)  # Randomly sample from -1, 0, and 1
    total_pts = np.sum(outcome)
    all_results[cur_sim] = total_pts

# Simulated p-value
print("Simulated p-value:")
simulated_p_value = np.mean(all_results >= my_point)
print(simulated_p_value)

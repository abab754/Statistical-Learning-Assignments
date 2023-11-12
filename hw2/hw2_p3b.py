import pandas as pd
import numpy as np

games = pd.read_csv("../premier.csv")
n_games = len(games)

teams = pd.unique(pd.concat([games.iloc[:, 0], games.iloc[:, 1]]))
n_teams = len(teams)


prior_probabilities = {
    'S': 0.4,
    'A': 0.3,
    'B': 0.2,
    'C': 0.1
}

ratings = {}
all_ratings = {team: [] for team in teams}

n_MCMC = 1100
n_Burn = 100

SAprob = 0.6
SBprob = 0.75
SCprob = 0.9
ABprob = 0.6
ACprob = 0.75
BCprob = 0.6
sameprob = 0.5

for team in teams:

    ratings[team] = np.random.choice(list(prior_probabilities.keys()), p=list(prior_probabilities.values()))

for j in range(n_MCMC):
    for cur_team in teams:
        weights = np.array([1.0] * n_teams)

        for i in range(n_games):
            if cur_team != games.iat[i, 0] and cur_team != games.iat[i, 1]:
                continue
            opponent_index = 1 if cur_team == games.iat[i, 0] else 0
            cur_index = 0 if opponent_index == 1 else 1
            cur_team_won = cur_index == 0
            opponent_team = games.iat[i, opponent_index]
            opponent_rating = ratings[opponent_team]


            if cur_team_won:
                if opponent_rating == 'S':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= SAprob
                elif opponent_rating == 'A':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= SBprob
                elif opponent_rating == 'B':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= SCprob
                elif opponent_rating == 'C':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= sameprob
            else:
                if opponent_rating == 'S':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= (1 - SAprob)
                elif opponent_rating == 'A':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= (1 - SBprob)
                elif opponent_rating == 'B':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= (1 - SCprob)
                elif opponent_rating == 'C':
                    team_index = np.where(teams == opponent_team)[0][0]
                    weights[team_index] *= (1 - sameprob)


        weights /= np.sum(weights)

        ratings[cur_team] = np.random.choice(teams, p=weights)

        if j > n_Burn:
            all_ratings[cur_team].append(ratings[cur_team])

for cur_team in teams:
    print(f"Posterior rating for {cur_team}")
    post_rate = pd.value_counts(all_ratings[cur_team], normalize=True).sort_index()
    print(post_rate)
    posterior_mean = np.sum(pd.to_numeric(post_rate.index, errors='coerce') * post_rate.values)
    print(f"Posterior mean rating: {posterior_mean}")
    print("----------")
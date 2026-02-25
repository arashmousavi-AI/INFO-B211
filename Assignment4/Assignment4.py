import pandas as pd
import numpy as np
import os
from scipy import stats, integrate, interpolate

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "players_stats_by_season_full_details.csv")

df = pd.read_csv(file_path)

print("th total rows:", len(df))
print("columns:", list(df.columns))


#1A) filtering only NBA regular season data

nba = df[(df["League"] == "NBA") & (df["Stage"] == "Regular_Season")].copy()

print("\nNBA regular season rows:", len(nba))
print("the unique stages in filtered data:", nba["Stage"].unique())


#1B) determining the player who played the MOST regular seasons
#counting unique season values per player
seasons_count = nba.groupby("Player")["Season"].nunique().sort_values(ascending=False)

top_player = seasons_count.index[0]
top_seasons = seasons_count.iloc[0]

print("\nPlayer with most NBA regular seasons:", top_player)
print("number of seasons:", top_seasons)


#1C)3-point accuracy per season for that player (accuracy = 3PM/3PA)
#Handling cases where 3PA might be 0 to avoid division by zero. We can set accuracy to NaN or 0 in those cases.
player_df = nba[nba["Player"] == top_player].copy()

player_df["3P_pct"] = np.where(player_df["3PA"] > 0,
                              player_df["3PM"] / player_df["3PA"],
                              np.nan)

print("\n3-point accuracy by season:")
print(player_df[["Season", "3PM", "3PA", "3P_pct"]].sort_values("Season"))


#converting season like "2002 - 2003" to numeric year 2002
#We'll use the first year as x-value. Based on my researches, this is a common convention for season-based data (for instance "2002 - 2003" is often referred to as the "2002 season").
player_df["Year"] = player_df["Season"].str.split(" - ").str[0].astype(int)

#sorting by year for the regression/interpolation
player_df = player_df.sort_values("Year")


#1D) the linear regression on 3P% across years
#droppinh NaNs in order avoid crashing
clean = player_df.dropna(subset=["3P_pct"])

x = clean["Year"].values
y = clean["3P_pct"].values

reg = stats.linregress(x, y)

print("\nlinear regression results:")
print("slope =", reg.slope)
print("intercept =", reg.intercept)
print("rvalue =", reg.rvalue)

#Here we calculat the line of best fit: y_hat = slope*x +intercept
def fit_line(t):
    return reg.slope * t + reg.intercept


#1E) average 3P% using integration of the fit line
#average value of f(x) on [a,b] is: (1 / (b-a)) * integrate_a**b f(x) dx

a = int(clean["Year"].min())
b = int(clean["Year"].max())

area, _ = integrate.quad(fit_line, a, b)
avg_3p_fit = area / (b - a)

actual_avg_3p = np.nanmean(player_df["3P_pct"]) #actual mean 3P%
actual_avg_3pm = np.nanmean(player_df["3PM"])   #actual mean 3PM

print("\nthe average 3P% from integrated best-fit line:", avg_3p_fit)
print("the actual average 3P% and mean of seasons:", actual_avg_3p)
print("Actual average 3-pointers made (3PM):", actual_avg_3pm)


# 1F)Interpolation to estimate missing seasons values (2002-2003 and 2015-2016)
# We'll estimate the missing 3P% using interpolation.
missing_years = [2002, 2015]  # from prompt seasons 2002-2003 and 2015-2016

interp_func = interpolate.interp1d(clean["Year"], clean["3P_pct"],
                                   kind="linear", fill_value="extrapolate")

estimated_vals = interp_func(missing_years)
print("\nestimated missing 3P% values (the liner interpolation):")
for yr, val in zip(missing_years, estimated_vals):
    print(f"Year {yr} -> estimated 3P% = {val}")



# 2A) Mean, variance, skew, kurtosis for FGM and FGA
fgm = nba["FGM"].dropna().values
fga = nba["FGA"].dropna().values

fgm_mean = np.mean(fgm)
fgm_var = np.var(fgm, ddof=1)  
fgm_skew = stats.skew(fgm, bias=False)
fgm_kurt = stats.kurtosis(fgm, bias=False)  

fga_mean = np.mean(fga)
fga_var = np.var(fga, ddof=1)
fga_skew = stats.skew(fga, bias=False)
fga_kurt = stats.kurtosis(fga, bias=False)

print("\nFGM stats:")
print("mean:", fgm_mean)
print("variance:", fgm_var)
print("skew:", fgm_skew)
print("kurtosis:", fgm_kurt)

print("\nFGA stats:")
print("mean:", fga_mean)
print("variance:", fga_var)
print("skew:", fga_skew)
print("kurtosis:", fga_kurt)


#2B)Relational t-test AND regular t-test
#Relational t-test = paired = same row relationship (FGM with FGA each season)
#Regular t-test = independent = treat them as separate groups

#For paired t-test, lengths must match and be aligned row-by-row:
paired_df = nba.dropna(subset=["FGM", "FGA"])
t_paired, p_paired = stats.ttest_rel(paired_df["FGM"], paired_df["FGA"])
t_ind, p_ind = stats.ttest_ind(fgm, fga, equal_var=False)

print("\npaired t-test (FGM vs FGA):")
print("t-stat =", t_paired)
print("p-value =", p_paired)

print("\nindependent t-test (FGM vs FGA):")
print("t-stat =", t_ind)
print("p-value =", p_ind)

#the interpretation using alpha=0.05 (the golden statistics number)
alpha = 0.05
print("\ninterpretation (alpha = 0.05):")
if p_paired < alpha:
    print("paired t-test:reject H0.")
else:
    print("paired t-test:fail to reject H0.")

if p_ind < alpha:
    print("Independent t-test:reject H0.")
else:
    print("Independent t-test:fail to reject H0.")
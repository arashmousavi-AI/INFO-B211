import os
import pandas as pd
import numpy as np
import csv


base_dir = os.path.dirname(os.path.abspath(__file__))
petal_file = os.path.join(base_dir, "Petal_Data.csv")
sepal_file = os.path.join(base_dir, "Sepal_Data.csv")

output_dir = "output"

# making the output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CSV files loading
petal_df = pd.read_csv(petal_file)
sepal_df = pd.read_csv(sepal_file)

# This part is optional and sanity check: make column names easier to match (lowercase, underscores)
petal_df.columns = [c.strip().lower().replace(" ", "_") for c in petal_df.columns]
sepal_df.columns = [c.strip().lower().replace(" ", "_") for c in sepal_df.columns]

# merging datasets into one DataFrame
df = pd.merge(petal_df, sepal_df, on=["sample_id", "species"], how="inner")

# keeping only required columns in the combined dataset
df = df[["sample_id", "species", "petal_length", "petal_width", "sepal_length", "sepal_width"]]

# making sure measurement columns are numeric and trying to turn bad strings into NaN. (Data cleaning steps)
cols = ["petal_length", "petal_width", "sepal_length", "sepal_width"] 
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# dropping rows with missing measurements
df = df.dropna(subset=cols)

# I wrote this code to save combined data
df.to_csv(os.path.join(output_dir, "iris_combined.tsv"), sep="\t", index=False)

# 6 correlations (all the species together)
corr_matrix = df[cols].corr()

pairs = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append([cols[i], cols[j], corr_matrix.loc[cols[i], cols[j]]])

corr_pairs_df = pd.DataFrame(pairs, columns=["var_1", "var_2", "correlation"])
corr_pairs_df.to_csv(os.path.join(output_dir, "correlations_overall.tsv"), sep="\t", index=False)


#mean, median, Std that are groupby species.
mean_df = df.groupby("species")[cols].mean()
median_df = df.groupby("species")[cols].median()
std_df = df.groupby("species")[cols].std()

# combining into one table 
stats_df = pd.concat(
    {"mean": mean_df, "median": median_df, "std": std_df},
    axis=1
)

stats_df.to_csv(os.path.join(output_dir, "stats_by_species.tsv"), sep="\t")


#most similar and least similar species
#using Euclidean distance between species mean vectors
species_means = mean_df  #already computed above
species_list = list(species_means.index)

dist_rows = []
for i in range(len(species_list)):
    for j in range(i + 1, len(species_list)):
        sp1 = species_list[i]
        sp2 = species_list[j]
        v1 = species_means.loc[sp1].values
        v2 = species_means.loc[sp2].values
        dist = np.sqrt(np.sum((v1 - v2) ** 2))
        dist_rows.append([sp1, sp2, dist])

dist_df = pd.DataFrame(dist_rows, columns=["species_1", "species_2", "euclidean_distance_between_means"])
dist_df = dist_df.sort_values("euclidean_distance_between_means", ascending=True)
dist_df.to_csv(os.path.join(output_dir, "species_similarity.tsv"), sep="\t", index=False)

# Printing quick summary
print("Total samples in combined dataset:", len(df))

if len(dist_df) > 0:
    most_sim = dist_df.iloc[0]
    least_sim = dist_df.iloc[-1]
    print("the most simila species:", most_sim["species_1"], "and", most_sim["species_2"])
    print("the least similar species:", least_sim["species_1"], "and", least_sim["species_2"])


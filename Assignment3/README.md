# Assignment 3 - Iris Data Analysis

## Purpose of the Program
The purpose of this program is to analyze the Fisher Iris dataset by combining petal and sepal measurements into one DataFrame and performing statistical analysis.

The program:
- Combines two CSV datasets into a single DataFrame

- Calculates correlations between measurement variables

- Computes mean, median, and standard deviation for each species

- Determines which iris species are most similar and least similar based on measurement averages

- Outputs the results for analysis

This project demonstrates how to perform heterogeneous data analysis using Pandas.

## Input:
The program takes the following inputs:
- Petal_Data.csv ---> contains petal length and petal width measurements
- Sepal_Data.csv ---> contains sepal length and sepal width measurements
Each dataset includes:
- Sample ID
- Species
- Measurement values
The program assumes both CSV files are located in the same directory as the Python file.

## Expected Output:
The program  prints:
- The total number of samples in the combined dataset.
- The most similar iris species.
- The least similar iris species.
Additionally, the program generates output files containing:
- Correlation matrix of the measurement variables
- Mean values for each species
- Median values for each species
- Standard deviation values for each of the species
These outputs helps comparing species based on the physical characterisitcs.

## Type of Execution
The program involves the following types of executions:
- 1. Sequential Execution: The code runs from top to bottom in a structured order.
- 2. Repeated Execution: The group-by operations apply statistical calculations across species groups.
- 3. Conditional Execution: The if statements are used when determining the most and least similar species based on the calculated distances.
- 4. Data Manipulation Executions: Pandas fundtion are heavily used to merge, group, calculate statistics, and compute the correlations.

## The Possible Improvements: 
The program could be improved by:
- Adding stronger error handling for the missing or incorrect formatted csv files.
- Allowing the users to choose which specific statistic to compute.
- Adding visualizations, such as charts or graph, in order to better compare species.


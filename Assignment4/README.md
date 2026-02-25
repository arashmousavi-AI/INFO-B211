# Assignment 4 - NBA Statistical Analysis

## Purpose of the Program
This project performs numerical data analysis on NBA player statistics using Python and the SciPy library. The dataset contains detailed player statistics by season.
The program filters NBA regular season data, analyzes three-point shooting trends, performs regression and interpolation, and applies statistical testing on shooting metrics.

The purpose of this program is to analyze NBA regular season statistics using statistical and numerical methods from SciPy.

The program:
- Filters the dataset to include only NBA regular season data
- Identifies the player with the most regular seasons played
- Calculates three-point shooting accuracy (3P%) for that player
- Performs linear regression on 3P% over time
- Estimates missing seasons using interpolation
- Calculates descriptive statistics (mean, variance, skew, kurtosis) for FGM and FGA
- Performs both paired and independent t-tests comparing FGM and FGA

This project demonstrates practical use of regression, integration, interpolation, and hypothesis testing on real-world sports data.

## Input:
The program takes the following inputs:
- A CSV file: players_stats_by_season_full_details.csv
Key columns used from the dataset:
- League
- Stage
- Season
- Player
- 3PM (3-point field goals made)
- 3PA (3-point field goals attempted)
- FGM (field goals made)
- FGA (field goals attempted)
No manual user input is required. The dataset is loaded directly using pandas.

## Expected Output:
The program outputs:
- Total number of NBA regular season records
- The player with the most regular seasons played
- That player’s three-point shooting accuracy per season
- Linear regression results (slope, intercept, correlation value)
- Average 3P% from integrated regression line
- Actual average 3P% and average 3PM
- Interpolated 3P% for missing seasons (2002–2003 and 2015–2016)
- Mean, variance, skewness, and kurtosis for FGM and FGA
- Results of paired and independent t-tests including p-values
- Interpretation of statistical significance at α = 0.05

All results are printed clearly to the console.

## Type of Execution
The program involves the following types of execution:

Sequential Execution:
The code executes in a linear top-to-bottom order.

Repeated Execution:
Grouping and aggregation operations iterate over players and seasons.

Conditional Execution:
If-statements are used to interpret t-test results based on p-values.

Numerical Computation:
SciPy functions are used for regression, integration, interpolation, and hypothesis testing.

Data Filtering:
Pandas is used to filter NBA regular season data and manipulate structured datasets.

## The Possible Improvements: 
The program could be improved by:
- Adding visualizations (for instance plotting regression line and shooting trends)
- Implementing better error handling for missing or corrupted CSV files
- Expanding analysis to compare multiple players instead of only one
- Performing polynomial regression instead of only linear regression
- Adding command-line arguments to allow selection of a specific player
- Creating a graphical dashboard for interactive exploration

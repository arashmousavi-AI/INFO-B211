# Assignment 2 – the NumPy NBA Season Analysis

## Purpose of the Program
This project analyzes NBA player performance data using Python and the NumPy library.  
The goal is to practice loading real-world data into the NumPy ndarrays and performing
numerical analysis to compute player and season-level statistics.

The analysis concentrates on shooting efficiency, scoring rates, and defensive performance
for the NBA players across different seasons.


## Data Source
The dataset used in this project is a CSV file containing season-level NBA player
statistics..


## Metrics Calculated
For each player and season, the program computes:

- Field goal accuracy
- Three-point accuracy
- Free throw accuracy
- Average points scored per minute
- Average points scored per game
- Overall shooting accuracy
- Average blocks per game
- Average steals per game


## Top 100 Rankings
For each metric listed above, the program identifies the top 100 player-season
performances and saves the results to separate .tsv files.

Each ranking includes:
- Player name
- Season
- Metric value


## Program Structure
The program is organized around a single analysis class that:
- Loads the CSV data into NumPy ndarrays
- Computes all required metrics using vectorized NumPy operations
- Exports results to tab-separated value (`.tsv`) files

Helper functions are used to simplify repeated operations such as safe division
and data exporting.


## Libraries Used
- **NumPy** – numerical computation and array manipulation  
- **os** – file and directory handling  
- **csv** – standard Python library used for structured data handling during export  


## How to Run the Program
1. Place the CSV data file in the same directory as the Python script.
2. Ensure NumPy is installed.
3. Run the script from the command line:

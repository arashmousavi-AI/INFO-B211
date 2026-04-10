# Assignment 8 - Data Visualization with Seaborn

## Purpose 
The purpose of this project is to perform data visualization using Python and the Seaborn library on real-world datasets. The project focuses on analyzing patterns and relationships in data through relational, distributional, and categorical visualizations.

Two datasets were used:

1. Seaborn Planets Dataset - to analyze astronomical features such as orbital period, mass, and discovery method
2. Exercise Dataset (Exercise_Data.csv) - to analyze the effect of diet and exercise on pulse rate

The goal is to transform raw data into meaningful insights through effective visualization techniques.

## Design and Implementation
The project is implemented using a modular, step-based structure rather than formal Python classes. Each section of the notebook acts as a logical component responsible for a specific task:
1. Data Loading
   - Loads datasets using pandas and seaborn
   - Performs initial inspection using .head(), .columns, and .isna()
2. Data Cleaning and Preprocessing
    - Handles missing values using .dropna()
   - Creates filtered subsets depending on visualization requirements
   - Applies transformations (like filtering extreme values, log scaling)
3. Data Visualization
   - Uses Seaborn functions (relplot, displot, catplot, heatmap)
   - Organizes plots into three categories:
      - Relational plots
      - Distribution plots
      - Categorical plots
     
## Data Transformation Module
- Converts datasets into appropriate formats for visualization
 - Example: Using pd.melt() to convert wide-format exercise data into long-format

## Attributes (Data Components)
The main attributes used in this project include:

Planets Dataset:
- orbital_period: Time taken for a planet to orbit its star
- mass: Mass of the planet
- method: Discovery method (e.g., Transit, Radial Velocity)
- year: Year of discovery
Exercise Dataset:
- diet: Type of diet (low fat, no fat)
- kind: Type of activity (rest, walking, running)
- pulse: Heart rate measurement
- time: Time interval (1 min, 15 min, 30 min)

## Methods Used
The following methods and functions were used throughout the project:

Data Handling:
- dropna() -> remove missing values
- groupby() -> aggregate data (e.g., discoveries per year)
- value_counts() -> identify most common categories
- melt() -> reshape data for categorical plotting

Visualization:
- sns.relplot() -> relational plots (scatter, line)
- sns.displot() -> distribution plots (histogram, KDE)
- sns.catplot() -> categorical plots (bar, box, violin)
- sns.heatmap() -> heatmap visualization

Plot Adjustments:
- plt.xscale("log"), plt.yscale("log") → handle skewed data
- plt.ylim() -> improve readability
- plt.xticks(rotation=...) -> enhance label clarity

## Key Implementation Decisions
- Log scaling was applied to variables like orbital period due to large value ranges
- Filtering was used to remove extreme outliers for better visualization clarity
- Top categories selection (like most common discovery methods) improved readability
- Long-format transformation enabled grouped categorical plots

## Limitations 
- Some columns (like planet mass) contain many missing values, reducing usable data
- Certain categories (like Transit method mass) have very limited variation
- Extreme outliers required filtering, which may remove some valid data points
- Overlapping points in scatter plots made some relationships difficult to interpret
- Swarm plots were not feasible due to dataset size and density

## Conclusion
This project demonstrates how data visualization techniques can be used to uncover meaningful insights from real-world datasets. By applying appropriate transformations such as filtering and log scaling, complex and highly skewed data can be effectively visualized.

The results show that:
- Planet discoveries have increased significantly over time
- Orbital period distributions vary greatly across discovery methods
- Exercise intensity has a stronger effect on pulse rate than diet

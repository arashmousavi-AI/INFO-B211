# Assignment 7 - Data Visualization with Matplotlib

## Purpose 
The purpose of this project is to perform data visualization using Python and Matplotlib on real-world datasets. The goal is to explore patterns, relationships, and trends within the Iris dataset and a Loan dataset by creating meaningful and interpretable visualizations.

## Design and Implementation
This project follows a structured, step-by-step approach rather than using object-oriented class design. The implementation is divided into logical sections, each responsible for a specific task:
1. Data Loading
   The datasets are loaded from CSV files into pandas DataFrames. This allows efficient manipulation and analysis of tabular data.
2. Data Cleaning and Preprocessing
   Certain columns, such as loan amounts, required cleaning due to formatting (e.g., currency symbols and commas). These values were converted into numeric format using pandas. Missing or invalid values were removed to ensure accurate and error-free visualizations.
3. Data Visualization
   Matplotlib was used to generate visualizations. Different types of plots were selected based on the type of data:
   - Scatter plots were used for comparing numeric variables (Iris dataset).
   - Histograms were used to examine distributions (interest rates).
   - Bar charts were used for comparing averages across categories (loan purpose and home ownership).
     
Each visualization was carefully designed with titles, axis labels, and grid lines to improve readability and interpretability.

## Key Attributes and Methods
Although no formal classes were implemented, the project relies on key attributes and methods provided by the pandas and matplotlib libraries:

- DataFrame Attributes (pandas)
  - Columns such as loan_amnt, loan_int_rate, loan_intent, and home_ownership are used as core attributes for analysis.
- Data Manipulation Methods (pandas)
  - read_csv() – loads datasets into DataFrames
  - groupby() – groups data by categorical variables
  - mean() – calculates average values
  - sort_values() – orders results for better visualization
  - dropna() – removes missing or invalid data
  - to_numeric() – converts data into numeric format
- Visualization Methods (matplotlib)
  - plt.scatter() – used for comparing two numeric variables
  - plt.hist() – used for displaying distributions
  - plt.bar() – used for comparing categorical data
  - plt.figure() – initializes plots
  - plt.grid() – improves readability
  - plt.tight_layout() – ensures proper spacing

These methods collectively enable efficient data processing and visualization.

## Iris Dataset Analysis
The visualizations of the Iris dataset reveal clear differences between species, particularly when comparing petal measurements. 
The scatter plots show that Setosa is distinctly separated from the other two species, while Versicolor and Virginica exhibit some overlap. 
Petal length and width provide stronger separation between species compared to sepal measurements. Additionally, the bar chart confirms differences
in average petal length across species. Overall, the visualizations demonstrate that certain features are more informative for classification tasks.

## Loan Dataset Analysis
The visualizations reveal several important patterns in the loan dataset. The distribution of interest rates shows that most loans fall within a moderate
range, with fewer loans at very high interest rates. The comparison of average loan amounts by loan purpose indicates that certain categories are associated 
with higher borrowing amounts, suggesting different financial needs across purposes. Additionally, the analysis of home ownership shows that borrowers with 
different housing situations request different loan amounts, with some groups consistently borrowing more than others. Overall, these results highlight how both
loan characteristics and borrower profiles influence borrowing behavior.

## Limitations 
This project has several limitations. First, some extreme outliers in the dataset were not fully analyzed, which may hide rare but important cases. Second, the analysis is 
based solely on visual interpretation and does not include statistical testing or predictive modeling. Finally, the project uses static visualizations, 
which limit interaction and deeper exploration of the data. Future improvements could include statistical analysis, interactive dashboards, or machine learning techniques to gain deeper insights.

## The Possible Improvements 
There are several ways this project could be improved in the future. One improvement would be to incorporate statistical analysis to validate the observed patterns and relationships. 
Another enhancement would be to use interactive visualization tools, such as Plotly, to allow users to explore the data more dynamically. Additionally, machine learning techniques
could be applied to predict loan behavior or classify Iris species more accurately. Finally, further data cleaning and handling of outliers could provide more robust and comprehensive insights.


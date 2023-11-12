
# Here's the documentation for the provided code in the notebook

## Linear Regression Analysis and Visualization
### Introduction
This code demonstrates a simple linear regression analysis and visualization using Python. It utilizes popular libraries such as pandas, numpy, matplotlib, and scikit-learn to import, analyze, and visualize data related to per capita income over the years.

### Required Packages
pandas (pd): Used for data manipulation and analysis.
numpy (np): Utilized for numerical operations and calculations.
matplotlib.pyplot (plt): Employed for data visualization.
sklearn.linear_model: Imported for linear regression modeling.

### Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


### Data Import
The code reads data from a CSV file called "canada_per_capita_income.csv" into a pandas DataFrame 'df' and displays the first few rows of the data for inspection.

#### Define the file path to the CSV file
path = "canada_per_capita_income.csv"
df = pd.read_csv(path)
df.head()

### Data Information
Summary information about the DataFrame 'df,' including data types, non-null values, and memory usage, is displayed to provide an overview of the dataset.
df.info()

### Data Visualization
The code uses Matplotlib to create a scatter plot of per capita income over the years. The plot is labeled with appropriate x and y-axis labels.

#### Set the label for the x-axis and y-axis
plt.xlabel("Years")
plt.ylabel("Per Capita Income ($)")

#### Create a scatter plot using the 'year' column as the x-axis and the 'per capita income (US$)' column as the y-axis
plt.scatter(df["year"], df["per capita income (US$)"], color="red", marker="+")


### Linear Regression Modeling
A linear regression model is created using the scikit-learn library. The model is trained with the 'year' column as the input feature and the 'per capita income (US$)' column as the target variable.

#### Create a LinearRegression object called 'reg'
reg = linear_model.LinearRegression()

#### Train the Linear Regression Model
reg.fit(df[["year"]], df["per capita income (US$)"])


### Linear Regression Equation
The linear regression equation used for predictions is displayed:

per_capita_income = 828.46507522 * year + (-1632210.7578554575)

### User Input and Prediction
The user is prompted to enter a year for which they want to predict per capita income. The code calculates and displays the predicted per capita income for the entered year using the linear regression equation.

#### Prompt the user to enter the year for prediction
year = float(input("Enter the year you want to predict:"))

#### Calculate the per capita income for the entered year
per_capita_income = 828.46507522 * year + (-1632210.7578554575)

#### Display the calculated per capita income for the entered year
print(f'The per capita income for the year {int(year)} is: ${per_capita_income:.2f}')


### Data Visualization with Regression Line
The code creates a figure with labels and a scatter plot for the actual data points. It also plots the linear regression line on the same graph and saves the visualization as an image.

#### Set the size of the figure
plt.figure(figsize=(16, 9))

####  Set the title and labels
plt.title("Visualization for Linear Regression One Variable", fontsize=20)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Per Capita Income ($)", fontsize=20)

####  Scatter plot for the actual data points
plt.scatter(df["year"], df["per capita income (US$)"], color="red", marker="+", label="Actual Data")

####  Plot the regression line
plt.plot(df["year"], reg.predict(df[["year"]]), color='blue', label="Regression Line")

####  Save the plot as an image
plt.savefig("LinearRegression_OneVariable.png")

####  Display the plot with a legend
plt.legend()
plt.show()

Conclusion
This code provides a complete example of performing linear regression analysis, visualizing the results, and making predictions based on the model. The linear regression model can be used to estimate per capita income for any given year within the range of the provided data.

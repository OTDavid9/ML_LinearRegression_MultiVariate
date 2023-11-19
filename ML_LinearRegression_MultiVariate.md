# LINEAR REGRESSION FOR MULTIPLE VARIABLES


# House Price Prediction

## Importing Relevant Packages


```python
# Ignore a specific warning (e.g., DeprecationWarning)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import the pandas library as 'pd' for data manipulation and analysis
import pandas as pd

# Import the numpy library as 'np' for numerical operations and calculations
import numpy as np

# Import the matplotlib.pyplot library as 'plt' for data visualization
import matplotlib.pyplot as plt

# Import linear_model from the scikit-learn (sklearn) library for machine learning
from sklearn import linear_model

# Print a message to indicate that the necessary packages have been imported
print("Packages imported!")

```

    Packages imported!
    

## Reading Dataset 


```python
# Define the file path to the CSV file containing the dataset
path = "homeprices.csv"

# Use pandas to read the data from the CSV file into a DataFrame named 'df'
df = pd.read_csv(path)

# Display the first few rows of the DataFrame to inspect the data
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>bedrooms</th>
      <th>age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>3.0</td>
      <td>20</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>4.0</td>
      <td>15</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>NaN</td>
      <td>18</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>3.0</td>
      <td>30</td>
      <td>595000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>5.0</td>
      <td>8</td>
      <td>760000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4100</td>
      <td>6.0</td>
      <td>8</td>
      <td>810000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display summary information about the DataFrame 'df,' including data types, non-null values, and memory usage
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 4 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   area      6 non-null      int64  
     1   bedrooms  5 non-null      float64
     2   age       6 non-null      int64  
     3   price     6 non-null      int64  
    dtypes: float64(1), int64(3)
    memory usage: 324.0 bytes
    

## Data Cleaning

Handling the missing values for bedrooms with median score


```python
median = df.bedrooms.median()

df.bedrooms.fillna(median, inplace = True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>bedrooms</th>
      <th>age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>3.0</td>
      <td>20</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>4.0</td>
      <td>15</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>4.0</td>
      <td>18</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>3.0</td>
      <td>30</td>
      <td>595000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>5.0</td>
      <td>8</td>
      <td>760000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4100</td>
      <td>6.0</td>
      <td>8</td>
      <td>810000</td>
    </tr>
  </tbody>
</table>
</div>



## Training LinearRegression Model


```python

# Select the independent variables (features) in 'X' by excluding the 'price' column
X = [col for col in df.columns if col != 'price']

# Define the dependent variable 'y' as the 'price' column
y = df['price']

# Create a LinearRegression model
reg = linear_model.LinearRegression()

# Fit the model using the selected independent variables 'X' and the dependent variable 'y'
reg.fit(df[X], y)

```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
# Use the trained 'reg' Linear Regression model to predict a target variable.
# Input features: 'Area', 'bedrooms', and 'age' for a specific data point.

# Gather input from the user
area = float(input("Enter the area of the house: "))  # Convert input to float
bedrooms = int(input("How many bedrooms does the house have? "))  # Convert input to int
age = int(input("How old is the house? "))  # Convert input to int

# Create an input feature list with the user-provided values
input_features = [[area, bedrooms, age]]

# Make the prediction using the trained model.
predicted_value = reg.predict(input_features)
predicted_value = int(predicted_value)
# Print the predicted value, which is the estimated outcome.
print("Predicted Price Value of the House is :", predicted_value)

```

    Enter the area of the house:  3000
    How many bedrooms does the house have?  3
    How old is the house?  40
    

    Predicted Price Value of the House is : 498408
    

    C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\base.py:464: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    


```python
# Access the coefficient (slope) of the linear regression model, which represents the relationship between the input feature and the target variable
reg.coef_

```




    array([  112.06244194, 23388.88007794, -3231.71790863])




```python
# Access the intercept (bias) term of the linear regression model, which represents the value of the target variable when the input feature is zero
reg.intercept_

```




    221323.0018654043



## Linear Regression Equation (MultiVariate)

### y = m1* x1 + m2 * x2 + m3*x3 +  b

### price =  area * 112.06244194 + bedrooms * 23388.88007794 + age *-3231.71790863 + 21323.0018654043


```python

```

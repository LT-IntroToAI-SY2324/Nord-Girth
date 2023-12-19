import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y

# Step 2: Standardize the data using StandardScaler, 

# Step 3: Transform the data

# Step 4: Split the data into training and testing data

# Step 5: Fit the data

# Step 6: Create a LogsiticRegression object and fit the data

# Step 7: Print the score to see the accuracy of the model

# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
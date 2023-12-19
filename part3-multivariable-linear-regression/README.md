# Multivariable Linear Regression

As in most decisions, there are more than one factor that goes into them, which is why we have multivariable linear regression.  We will be using multiple variables to decide on our regressions.

Complete the examples in class with Mr. Berg, then complete `a6_part3.py` on your own and answer the questions in `a6_part3_writeup.md`

## Create the Model
Create a multivariable linear regression model that describes the relationship between a car’s mileage, age, and price.

1. Create the Model
- Use the train_test_split function to split the dataset into training data and testing data.
- Find and print out the coefficients, bias, and R2 values.
2. Test the Model
- Loop through the values in the xtest dataset and print the predicted and actual values to the console.
- Note you should print out both of the x values (the miles and the age) that go into each predicted y value (the price of the car).

## Analyze the Data
Once you have all this information about your model, answer the following questions in the next exercise:

1. What is the R Squared coefficient for your model? What does that mean about the model in relation to your data?

2. Is your model accurate? Why or why not?

3. What does the model predict a 10-year-old car with 89000 miles is worth? What about a car that is 20 years old with 150000 miles?

4. You may notice that some of your predicted results are negative. This is occurring when the value of age and the mileage of the car are very high. Why do you think this is happening?
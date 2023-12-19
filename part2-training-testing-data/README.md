# Creating Predictive Models - Training and Testing Data

Before starting the assignment, Mr. Berg will go over `example1-crickets.py`.  In this example we will explore training and testing data and discuss the importance of how we train our models and then test on solutions that we know.

After completing that in class you will complete `a6_part2.py` where you will be completing the same example from part 1, but splitting the data appropriately.  Once finished, make sure to complete the `part2_writeup.md` and save before pushing to github.

## Create the Model

In this exercise, you will build on the linear regression model that you created in the previous lesson for the blood pressure and age dataset. To make your model more effective, you will create a testing and training dataset.

Complete the following steps:

1. Create the Model
- Use the train_test_split function to split the dataset into training data and testing data.
- Reshape the xtrain values to be a 2D array.
- Determine the coefficient, bias, and R2 values.
- Print the model’s linear equation and R2 value.
2. Test the Model
- Loop through the values in the xtest dataset and print the predicted and actual values to the console.
Note: You do not need to create a visual of the model.

Once you have a functioning program, run the program a few times. Notice that your values will change because the train_test_split function uses a randomized set each time the program is run.

## Analyze the Data
Once you have successfully created a program that models the relationship between blood pressure and age, use the following questions to analyze the data. You will answer these questions in the next activity.

- What makes this model more effective than the model you created in the previous lesson?
- What does the R squared coefficient tell you about the model?
Would you say that your model is accurate? What evidence supports your conclusion? Consider the meaning of the predicted and actual values in the context of the chart below from the American Heart Association’s website on understanding blood pressure.

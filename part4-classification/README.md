# Classifications

Before completing this assignment, make sure you fully understand the Iris Classification problem example.

## SUV Data

A car company wants a better sense of which customers are going to buy their products. They’ve collected data about people who walk into their store, and whether or not they purchased an SUV from their store.
They want to know based on a persons age, gender and income if they will buy a car. Your job is to create a logistic regression model to help them make this prediction.

The x and y values have been created for you - before starting, you should print out their values to make sure you understand what data you are including in your model. You’ll notice that gender has been transformed into a 0 or 1 value. This is so that it can be used in our model.

Before creating the model, you will have to standardize the data. This means creating a StandardScaler and fitting the independent variables to the scaler. Then, transform the data so that it fits the correct dimensions. Consider printing this to see how the data has been reformatted.

Once the data is scaled, create a LogsiticRegression object and fit the data. Then, print out the score of the testing data to see the accuracy of the model.

Print out the actual ytest values and predicted y values based on the xtest data, and compare them. Answer the following questions in the next exercise:

Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?

1. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.

2. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?

3. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.


Now that you’ve chosen your dataset, it’s time to choose which algorithm you will use. Throughout this module, you have studied the following algorithms:

- Linear regression
- Logistic regression (classification)
- K-means (clustering)
One of the most challenging parts of mastering machine learning is choosing the right algorithm. This decision should be guided by the dataset as well as your goal for analyzing the data:

- Is there a linear correlation between variables?
- Do you want to predict a numerical value?
- Are you wanting help structuring the data into groups?
These are the types of questions you will need to ask yourself as you choose which algorithm is best for your chosen dataset.

The first step is to get familiar with your data. What variables are included in the dataset? What is the driving question or purpose of your model? Once you have an understanding of these two aspects of your dataset, you can start the process of choosing the right algorithm.


<b>Supervised vs. Unsupervised Learning Algorithm</b>

When choosing between an unsupervised and supervised algorithm, there is one key question to consider: Do you know the label, or outcome, for your output? For example, when you were working with the car prices dataset, you knew the outcome you were looking for was the price of the car. When you looked at the SUV dataset, you knew the outcome you were looking for a specific category: will buy an SUV and will not buy an SUV. If this is the case, you will want to use a supervised learning algorithm.

On the other hand, when you analyzed the mall customer data, you did not know the label for output. Instead, the computer created the groups for you and you created a label based on the computer-generated groups. If this is the case, then will you want to use an unsupervised learning algorithm.

<b>Linear vs. Logistic Regression</b>
If you decided on a supervised learning algorithm, the next step is to identify if you should use linear or logistic regression. Linear regression models are used for predicting numerical values, such as the price of a car or the time needed to travel between two places. Logistic regression models are used for classification, or when you want to label each output as part of a specific group.
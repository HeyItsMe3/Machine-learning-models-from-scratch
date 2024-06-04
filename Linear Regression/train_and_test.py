from linear_regression import LinearRegression
import numpy as np

# test the model
""" 
Dataset: Advertising Campaigns

Description: A company wants to predict the effectiveness of their advertising campaigns based on the features of the ads and the response they receive.

Features:

1. TV Ad Spend (thousands of dollars): The amount spent on TV ads.
2. Online Ad Impressions (thousands of impressions): The number of times the company's ads were shown online.

Training Data (8 rows):

| TV Ad Spend | Online Ad Impressions | Sales |
| --- | --- | --- |
| 2.1 | 15.3 | 8.5 |
| 4.8 | 20.1 | 19.2 |
| 3.2 | 18.9 | 12.1 |
| 5.4 | 22.7 | 22.9 |
| 1.9 | 14.1 | 7.3 |
| 3.5 | 19.5 | 14.1 |
| 4.9 | 21.3 | 20.5 |
| 2.3 | 17.9 | 9.7 |

Testing Data (2 rows):

| TV Ad Spend | Online Ad Impressions | Sales |
| --- | --- | --- |
| 3.7 | 18.1 | ? |
| 5.1 | 24.5 | ? |

results should be: 14.9 and 20.2

Task: Use the training data to train a linear regression model that predicts the sales based on the TV ad spend and online ad impressions. Then, use the testing data to test the accuracy of the model.

 """

model = LinearRegression()
features = 2
initial_weight = [0,1,0] # define the initial weights
tv_add_spend = [2.1,4.8,3.2,5.4,1.9,3.5,4.9,2.3]
online_ad_impressions = [15.3,20.1,18.9,22.7,14.1,19.5,21.3,17.9]
sales = [8.5,19.2,12.1,22.9,7.3,14.1,20.5,9.7]
x_train = [[1,1,1,1,1,1,1,1],tv_add_spend, online_ad_impressions]
y_train = sales

iteration = 1000
learning_rate = 0.00535 # choose the best learning rate
w = model.gradient_descent(initial_weight,x_train,y_train,iteration,learning_rate)
print(w)
x_test = [[1,1],[3.7,5.1],[18.1,24.5]]
prediction = model.hypothesis(w,x_test)
print(prediction)

# Expected Output: [14.52, 19.35]
percent_error = np.abs((prediction[0]-14.9)/14.9)*100
print(f"Accuracy: {100-percent_error}%")




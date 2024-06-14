from logistic_regression_regularised import LogisticRegressionRegularised
import numpy as np

# test the model
""" 
**Features:**

| Feature | Description |
| --- | --- |
| Age | Patient's age in years |
| BMI | Patient's body mass index |
| Smoking | 0 = non-smoker, 1 = smoker |
| Family History | 0 = no family history, 1 = family history |
| Cholesterol Level | Patient's cholesterol level |

**Target:**

| Target | Description |
| --- | --- |
| Heart Disease | 0 = no heart disease, 1 = heart disease |

**Data:**

| Age | BMI | Smoking | Family History | Cholesterol Level | Heart Disease |
| --- | --- | --- | --- | --- | --- |
| 45 | 25 | 1 | 1 | 200 | 1 |
| 60 | 30 | 0 | 1 | 220 | 1 |
| 35 | 22 | 1 | 0 | 180 | 0 |
| 55 | 28 | 1 | 1 | 240 | 1 |
| 40 | 25 | 0 | 1 | 200 | 0 |
| 65 | 32 | 1 | 1 | 260 | 1 |

test data to predict the heart disease
| 38 | 24 | 1 | 0 | 190 | 0 |
| 52 | 29 | 1 | 1 | 220 | 1 |

 """

model = LogisticRegressionRegularised()


total_features = 5
weights = np.ones(total_features+1)
age = [45,60,35,55,40,65]
bmi = [25,30,22,28,25,32]
smoking = [1,0,1,1,0,1]
family_history = [1,1,0,1,1,1]
cholestrol_level = [200,220,180,240,200,260]
heart_disease = [1,1,0,1,0,1]

x_train = [[1,1,1,1,1,1],age,bmi,smoking,family_history,cholestrol_level]
y_train = heart_disease
iteration = 20
learning_rate = 0.01

""" w = model.gradient_descent(weights,x_train,y_train,iteration,learning_rate)
#print(model.gradient_descent(weights,x,y,iteration,learning_rate))
#print(model.sigmoid_function([5.12,28.09],x=[[1],[11]]))
x_test = [[1,1],[38,52],[24,29],[1,1],[0,1],[190,220]]
prediction = model.sigmoid_function(w,x_test)
for result in prediction:
    print(round(result))
 """
x_test = [[1,1],[38,52],[24,29],[1,1],[0,1],[190,220]]

model.gradient_descent(weights,x_train,y_train,iteration,learning_rate, lambda_=0.5)
a = model.train(x_train,y_train,lambda_=0.5,weights=weights,iteration=iteration,learning_rate=learning_rate)
print(model.predict(a,x_test))
#Linear Regression

#Step1:Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Step2:Reading data from url using pandas

url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data=pd.read_csv(url)
print(s_data.head(10))
print(s_data.tail(10))

#Step3:Plotting Distribution of scores where x-cor represents hours and y-cor represents scores

s_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours Vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scores')
plt.show()
#Step4:Dividing the data into dependent and independent features

X=s_data.iloc[:,:-1].values
y=s_data.iloc[:,1].values
print(X)
print(y)

#Step5:Splitting the data into training and testing dataset

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(f"X_train\n{X_train}\n X_test\n{X_test}")

#Step6:Training the algorithm

regressor=LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train)
print("Training completed")

#Plotting the regression line

line=regressor.coef_*X+regressor.intercept_

#Plotting for test dataset

plt.scatter(X,y)
plt.plot(X,line,color='red')
plt.show()

#Making Predictions

y_pred=regressor.predict(X_test)

#Comparing the actual and predicted values

df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)

#What will be predicted score if a student studies for 9.25hrs/day?

hours=9.25
own_pred= regressor.predict([[hours]])
print(f"No of Hours={hours}")
print(f"Predicted Scores={own_pred[0]}")


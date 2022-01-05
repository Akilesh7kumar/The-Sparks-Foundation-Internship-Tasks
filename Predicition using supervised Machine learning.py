
# Author: AKILESH KUMAR R 


# Task 1 : Prediction using Supervised Machine Learning

# GRIP @ The Sparks Foundation

# Step 1: Importing all the libraries and reading the data

#Importing all the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head()

# Step 2:Data Visualization

data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('No of Hours Studied')  
plt.ylabel('Percentage Scored')  
plt.show()

# Step 3: Data Preprocessing

data.corr()

data.isnull()

x = data.iloc[:,:-1]
y = data.iloc[:,1]

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0) 

# Step 4: Model Training

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

y_pred=regressor.predict(X_test)

y_pred

# Step 5: Regression Line plotting

line = regressor.coef_*x+regressor.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line,color='red');
plt.show()

# Step 6: Predictions

# You can also test with your own data
hours = 9.25
test  = np.array([hours])
test=test.reshape(-1,1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

df = pd.DataFrame({'Actual':y_test , 'Predicted':y_pred })

df

# Step 7: Model Evaluation

from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))

# Conclusion:
          

   #          I carried out the task sucsessfully and evaluated the performance measures
   
   

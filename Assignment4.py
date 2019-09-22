import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#Import the housing dataset
df_2 = pd.read_csv("housing2.csv") #With the 13 noise attributes
df_1 = pd.read_csv("housing.csv")

#Clean up the data set. Remove rows where one entry is NaN
df_1.replace(["Nan"],np.nan,inplace=True)
df_1 = df_1.dropna()
df_2.replace(["Nan"],np.nan,inplace=True)
df_2 = df_2.dropna()

# Linear Regression. 
X = df_1.iloc[0:,:13]
y = df_1.iloc[0:,13]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_test_pred = reg.predict(X_test)
y_train_pred = reg.predict(X_train)

print('Intercept: %.3f' % reg.intercept_)
print("Coefficients: " + str(reg.coef_))


print('MSE train: %.3f, test: %.3f' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

# Residual errors plot:
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

# Ridge regression
for i in [0.2,0.4,0.6,0.8,1.0]:
    print("For alpha = " + str(i))
    ridge = Ridge(alpha=i)
    ridge.fit(X_train, y_train)
    y_test_pred = ridge.predict(X_test)
    y_train_pred = ridge.predict(X_train)
    print('Intercept: %.3f' % ridge.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
    print("Coefficients: " + str(ridge.coef_))
    
# Residual errors plot:
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
print("For this residual error plot, alpha = 1")
plt.show()

# Lasso Regression
for i in [0.01,0.2,0.4,0.6,0.8,1.0]:
    print("For alpha = " + str(i))
    lasso = Lasso(alpha=i)
    lasso.fit(X_train, y_train)
    y_test_pred = lasso.predict(X_test)
    y_train_pred = lasso.predict(X_train)
    print('Intercept: %.3f' % lasso.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
    print("Coefficients: " + str(lasso.coef_))
    
# Residual errors plot:
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
print("For this residual error plot, alpha = 1")
plt.show()

# Lasso Regression with the 13 noise attributes
X = df_2.iloc[0:,:26].values
y = df_2.iloc[0:,26].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

for i in [0.01,0.2,0.4,0.6,0.8,1.0]:
    print("For alpha = " + str(i))
    lasso = Lasso(alpha=i)
    lasso.fit(X_train, y_train)
    y_test_pred = lasso.predict(X_test)
    y_train_pred = lasso.predict(X_train)
    print('Intercept: %.3f' % lasso.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
    print("Coefficients: " + str(lasso.coef_))
    
# Residual errors plot:
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
print("For this residual error plot, alpha = 1")
plt.show()

# Ridge regression
for i in [0.2,0.4,0.6,0.8,1.0]:
    print("For alpha = " + str(i))
    ridge = Ridge(alpha=i)
    ridge.fit(X_train, y_train)
    y_test_pred = ridge.predict(X_test)
    y_train_pred = ridge.predict(X_train)
    print('Intercept: %.3f' % ridge.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
    print("Coefficients: " + str(ridge.coef_))
    
# Residual errors plot:
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
print("For this residual error plot, alpha = 1")
plt.show()

# Exploratory data analysis (part 1)
print("Summary of our dataset.(Without the noise attributes)")
print("Shape of our dataset: "  + str(df_1.shape))
print(df_1.head())
print("Sumary statistics of the data from our dataset:" )
print(df_1.describe())

# Heat map
print("Correlation matrix:")
cm = np.corrcoef(df_1[:].values.T)
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(10,10))
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 10},yticklabels=list(df_1.columns)
                 ,xticklabels=list(df_1.columns),ax=ax)
plt.show()

plt.figure()
sns.set(style='whitegrid',context = 'notebook')
sns.pairplot(df_1[:],size=2.5)
plt.show

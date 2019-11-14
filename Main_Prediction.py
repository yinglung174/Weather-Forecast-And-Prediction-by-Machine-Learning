#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import wx
import os
import time
import sys







# In[ ]:


input_max_temp = input("Please input maximum of temperature: ")
input_min_temp = input("Please input minimum of temperature: ")
input_meandew = input("Please input mean dew point: ")
input_meanhum = input("Please input mean humidity: ")
input_pressure = input("Please input mean pressure: ")
input_meancloud = input("Please input mean cloud: ")
input_rainfall = input("Please input mean rainfall: ")
input_population = input("Please input population density: ")
input_sunshine = input("Please input mean number of sunshine hour: ")
input_wind_dir = input("Please input mean wind direction: ")
input_wind_speed = input("Please input mean wind speed: ")
input_air_quality = input("Please input mean air health quality: ")

# In[ ]:


if (True):
    # !/usr/bin/env python
    # coding: utf-8

    # In[1]:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as seabornInstance
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn import metrics

    # In[2]:

    dataset = pd.read_csv('data_2.csv')

    # In[3]:

    dataset.shape

    # In[4]:

    dataset.describe()

    # In[5]:

    dataset.isnull().any()

    # In[6]:

    dataset = dataset.fillna(method='ffill')

    # In[7]:

    dataset.plot(x='pressure', y='mean_temp', style='o')
    plt.title('Pressure vs Mean Temperature')
    plt.xlabel('pressure')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/pressure.png")
    plt.show()
    dataset.plot(x='max_temp', y='mean_temp', style='o')
    plt.title('Maximum Temperature vs Mean Temperature')
    plt.xlabel('max_temp')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/max_temp.png")
    plt.show()
    dataset.plot(x='min_temp', y='mean_temp', style='o')
    plt.title('Minimum Temperature vs Mean Temperature')
    plt.xlabel('min_temp')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/min_temp.png")
    plt.show()
    dataset.plot(x='meandew', y='mean_temp', style='o')
    plt.title('Mean Dew Point vs Mean Temperature')
    plt.xlabel('meandew')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/meandew.png")
    plt.show()
    dataset.plot(x='meanhum', y='mean_temp', style='o')
    plt.title('Mean Humidity vs Mean Temperature')
    plt.xlabel('meanhum')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/meanhum.png")
    plt.show()
    dataset.plot(x='meancloud', y='mean_temp', style='o')
    plt.title('Mean Cloud vs Mean Temperature')
    plt.xlabel('meancloud')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/meancloud.png")
    plt.show()
    dataset.plot(x='rainfall', y='mean_temp', style='o')
    plt.title('Rainfall vs Mean Temperature')
    plt.xlabel('rainfall')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/rainfall.png")
    plt.show()
    dataset.plot(x='population', y='mean_temp', style='o')
    plt.title('Population vs Mean Temperature')
    plt.xlabel('Population')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/population.png")
    plt.show()
    dataset.plot(x='sunshine_hour', y='mean_temp', style='o')
    plt.title('Bright Sunshine vs Mean Temperature')
    plt.xlabel('sunshine_hour')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/sunshine.png")
    plt.show()
    dataset.plot(x='wind_direction', y='mean_temp', style='o')
    plt.title('Wind Direction vs Mean Temperature')
    plt.xlabel('wind_direction')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/wind_direction.png")
    plt.show()
    dataset.plot(x='wind_speed', y='mean_temp', style='o')
    plt.title('Wind Speed vs Mean Temperature')
    plt.xlabel('wind_speed')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/wind_speed.png")
    plt.show()
    dataset.plot(x='air_health_quality', y='mean_temp', style='o')
    plt.title('Air Health Quality vs Mean Temperature')
    plt.xlabel('air_health_quality')
    plt.ylabel('mean_temp')
    plt.savefig("statistics/air_quality.png")
    plt.show()

    # In[8]:

    X = dataset[['pressure', 'max_temp', 'min_temp', 'meandew', 'meanhum', 'meancloud', 'rainfall', 'population',
                 'sunshine_hour', 'wind_direction', 'wind_speed', 'air_health_quality']]
    y = dataset['mean_temp']

    # In[9]:

    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    seabornInstance.distplot(dataset['mean_temp'])

    # In[10]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # In[11]:
    print("Linear Regression Prediction: ")

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # In[12]:

    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    coeff_df.sort_values(by='Coefficient', ascending=False)

    # In[13]:

    pos_coeffs_df = coeff_df[(coeff_df['Coefficient'] >= 0)].sort_values(by='Coefficient', ascending=False)
    # pos_coeffs_df.sort_values(by='Estimated_Coefficients', ascending=False)
    pos_coeffs_df

    # In[14]:

    pos_coeffs_df = coeff_df[(coeff_df['Coefficient'] < 0)].sort_values(by='Coefficient', ascending=True)
    # pos_coeffs_df.sort_values(by='Estimated_Coefficients', ascending=False)
    pos_coeffs_df

    # In[15]:

    y_pred = regressor.predict(X_test)

    # In[16]:

    import seaborn as sns

    g = sns.regplot(y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Predicted Mean Temperature', ylabel='Actual Mean Temperature', title='Model Predictions')
    plt.title('Regression Plot for Actual vs Predicted Values')

    # In[17]:

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1

    # In[18]:

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("statistics/linear_regression_comparison.png")
    plt.show()

    # In[19]:

    # R2 for train and test data
    R2_reg_train = regressor.score(X_train, y_train)
    R2_reg_test = regressor.score(X_test, y_test)
    print('R squared for train data is: %.3f' % (R2_reg_train))
    print('R squared for test data is: %.3f' % (R2_reg_test))

    # In[20]:

    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - regressor.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - regressor.predict(X_test)) ** 2))
    print('Root mean squared error for train data is: %.3f' % (RMSE_reg_train))
    print('Root mean sqaured error for test data is: %.3f' % (RMSE_reg_test))

    # In[21]:

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # In[27]:

    # input_pressure = 1000
    # input_max_temp = 30
    # input_min_temp = 25
    # input_meandew = 25
    # input_meanhum = 80
    estimated_temp = regressor.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum),float(input_meancloud),float(input_rainfall),int(input_population),float(input_sunshine),float(input_wind_dir),float(input_wind_speed),float(input_air_quality)]])
    print ("The expected mean of temperature is", estimated_temp)

    print(" ")
    print("K-Nearest Neighbors Prediction: ")

    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)

    # In[12]:

    pred_knn = knn.predict(X_test)
    pred_knn
    y_pred = knn.predict(X_test)

    # In[16]:

    import seaborn as sns

    g = sns.regplot(y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Predicted Mean Temperature', ylabel='Actual Mean Temperature', title='Model Predictions')
    plt.title('Regression Plot for Actual vs Predicted Values')

    # In[17]:

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1

    # In[18]:

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("statistics/KNN_comparison.png")
    plt.show()

    # In[19]:

    # R2 for train and test data
    # R2 for train and test data
    R2_reg_train = knn.score(X_train, y_train)
    R2_reg_test = knn.score(X_test, y_test)
    print('R squared for train data is: %.3f' % (R2_reg_train))
    print('R squared for test data is: %.3f' % (R2_reg_test))

    # In[20]:

    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - knn.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - knn.predict(X_test)) ** 2))
    print('Root mean squared error for train data is: %.3f' % (RMSE_reg_train))
    print('Root mean sqaured error for test data is: %.3f' % (RMSE_reg_test))

    # In[21]:

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # In[27]:

    # input_pressure = 1000
    # input_max_temp = 30
    # input_min_temp = 25
    # input_meandew = 25
    # input_meanhum = 80
    estimated_temp = knn.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum),float(input_meancloud),float(input_rainfall),int(input_population),float(input_sunshine),float(input_wind_dir),float(input_wind_speed),float(input_air_quality)]])

    print ("The expected mean of temperature is", estimated_temp)

# In[ ]:

    print(" ")
    print("Random Forest Regression Prediction: ")

    rf = RandomForestRegressor(random_state=5, n_estimators=20)
    rf.fit(X_train, y_train)

    # In[12]:

    pred_rf = rf.predict(X_test)
    pred_rf
    y_pred = rf.predict(X_test)

    # In[16]:

    import seaborn as sns

    g = sns.regplot(y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Predicted Mean Temperature', ylabel='Actual Mean Temperature', title='Model Predictions')
    plt.title('Regression Plot for Actual vs Predicted Values')

    # In[17]:

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1

    # In[18]:

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("statistics/random_forest_comparison.png")
    plt.show()

    # In[19]:

    # R2 for train and test data
    # R2 for train and test data
    # R2 for train and test data
    R2_reg_train = rf.score(X_train, y_train)
    R2_reg_test = rf.score(X_test, y_test)
    print('R squared for train data is: %.3f' % (R2_reg_train))
    print('R squared for test data is: %.3f' % (R2_reg_test))

    # In[20]:

    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - rf.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - rf.predict(X_test)) ** 2))
    print('Root mean squared error for train data is: %.3f' % (RMSE_reg_train))
    print('Root mean sqaured error for test data is: %.3f' % (RMSE_reg_test))

    # In[21]:

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # In[27]:

    estimated_temp = rf.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum),float(input_meancloud),float(input_rainfall),int(input_population),float(input_sunshine),float(input_wind_dir),float(input_wind_speed),float(input_air_quality)]])

    print ("The expected mean of temperature is", estimated_temp)


# In[ ]:





# In[ ]:





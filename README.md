# Weather-Forecast-And-Prediction-by-Machine-Learning




**

## Background

**
For the current situation, Hong Kong observatory conduct a traditional weather forecasting. There are four common methods to predict weather. The first method is climatology method that is reviewing weather statistics gathered over multiple years and calculating the averages.The second method is analog method that is to find a day in the past with weather similar to the current forecast. The third method is persistence and trends method that has no skill to predict the weather because it relies on past trends. The fourth method isnumerical weather prediction the is making weather predictions based on multiple conditions in atmosphere such as temperatures, wind speed, high-and low-pressure systems, rainfall, snowfall and other conditions.So,there are many limitations of these traditional methods. Not only It forecasts the temperature in the current month at most, but also it predicts without using machine learning algorithms.Therefore, my project is to increase the accuracy and predict weather in the future at least one month through applying machine learning techniques.

**

## Objective (Brief)

**
There are two purposes of my project. One of the purposeis to forecast the status of weather in the August of specific year. I will demonstrate the result through using decision tree regression and show the output for the status of wet or heat. Another aim is to predict the temperature using different algorithms like linear regression, random forest regression and K-nearest neighbor regression. The output value should be numerical based on multiple extra factors like population density and air health quality.

![enter image description here](https://lh3.googleusercontent.com/43WkMUHGBC12Fap74eYDH-rsIg7BgmaeAev2f_xhoa1hg678kmiQbIEawUfKkjOjsrvpzhzUIvy9 "experiment")

**

## Purpose (Detail)

**
To forecast the status of weather in the August of next year
**ML Algorithm**: Decision Tree Regression
**Status**: wet and heat 
**Output Value**: Yes / No
To predict the temperature using Different Algorithms
**ML Algorithms**: Linear Regression,
	 	Random Forest Regression, K-Nearest Neighbor
**Output Value**: Numerical

**Algorithm - Decision Tree:**  builds regression or classification models in the form of a tree structure
**Algorithm - Linear Regression:** performs the task to predict a dependent variable value (y) based on a given independent variable (x)
**Algorithm - Random Forest Regression:** performing both regression and classification tasks using multiple decision trees and a statistical technique called bagging
**Algorithm - K-Nearest Neighbor Regression**

**Data Source:** Hong Kong Observatory, aqhi.gov.hk
**Dynamic Data:** August of 1999 - 2019
**Static Data:** June of 2014 - 2019

**

## Data Description:

**
**mean_temp:** mean air temperature
**max_temp:** mean daily maximum air temperature
**min_temp:** mean daily minimum air temperature
**meanhum:** mean relative humidity
**meandew:** mean dew point temperature
**pressure:** mean daily air pressure
**heat:** true when mean air temperature is over or equal to 30
**wet:** true when mean relative humidity is over or equal to 80
**Mean_cloud:** mean cloud
**population:** population density
**Sunshine_hour:** mean number of hour of sunshine
**Wind_direction:** mean wind direction
**Wind_speed:** mean wind speed
**Air_health_quality:** mean daily air health quality

**

## System Requirement:

**
Python 3.6
BeautifulSoup
Pandas
Numpy
Matplotlib
Seaborn
Openpyxl
Sklearn
wxPython

**

## Function of Program:

**
**‘Forecast’ Button**: Forecast the status of weather in the August of next year
**‘Activate Auto-Forecast’ Button**: Periodically forecast the status of weather
**‘Prediction’ Button**: Predict the mean temperature based on other factors

**

## User Guide:

**
**Step 1**: Download & Install Python 3.6
**Step 2**: Go to Terminal & Download Python Library (py -3.6 –m pip install ____)
**Step 3**: Go to ‘Weather_Prediction’ folder & click GUI.cpython-36
**Step 4**: The forecast result will be stored in ‘prediction’ folder. The prediction statistics will be stored in ‘statistics’ folder

**

## Developer Tools:

**
**Programming Language**: Python
**IDE**: PyCharm
**GUI**: wxPython, wxFormBuilder
**Web Scraping**: BeautifulSoup, ParseHub
**Debugging & Testing**: Jupyter Notebook
**Data Format**: Microsoft Excel

```

# Weather-Forecast-And-Prediction-by-Machine-Learning

![image](https://user-images.githubusercontent.com/37294801/83221305-09402d80-a1a8-11ea-83e4-79625d909a15.png)
![image](https://user-images.githubusercontent.com/37294801/83221317-11986880-a1a8-11ea-914e-9747e9c71066.png)
![image](https://user-images.githubusercontent.com/37294801/83221323-152bef80-a1a8-11ea-837b-492af14a426f.png)
![image](https://user-images.githubusercontent.com/37294801/83221359-32f95480-a1a8-11ea-86a3-8afbf754a2af.png)
![image](https://user-images.githubusercontent.com/37294801/83221420-5ae8b800-a1a8-11ea-9aa5-755cc495aa57.png)
![image](https://user-images.githubusercontent.com/37294801/83221401-502e2300-a1a8-11ea-8639-19284354128f.png)


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
To forecast the status of weather in the August of next year<br/>
**ML Algorithm**: Decision Tree Regression<br/>
**Status**: wet and heat <br/>
**Output Value**: Yes / No<br/>
To predict the temperature using Different Algorithms<br/>
**ML Algorithms**: Linear Regression,
	 	Random Forest Regression, K-Nearest Neighbor<br/>
**Output Value**: Numerical<br/>

**Algorithm - Decision Tree:**  builds regression or classification models in the form of a tree structure<br/>
**Algorithm - Linear Regression:** performs the task to predict a dependent variable value (y) based on a given independent variable (x)<br/>
**Algorithm - Random Forest Regression:** performing both regression and classification tasks using multiple decision trees and a statistical technique called bagging<br/>
**Algorithm - K-Nearest Neighbor Regression**<br/>

**Data Source:** Hong Kong Observatory, aqhi.gov.hk<br/>
**Dynamic Data:** August of 1999 - 2019<br/>
**Static Data:** June of 2014 - 2019<br/>

**

## Data Description:

**
**mean_temp:** mean air temperature<br/>
**max_temp:** mean daily maximum air temperature<br/>
**min_temp:** mean daily minimum air temperature<br/>
**meanhum:** mean relative humidity<br/>
**meandew:** mean dew point temperature<br/>
**pressure:** mean daily air pressure<br/>
**heat:** true when mean air temperature is over or equal to 30<br/>
**wet:** true when mean relative humidity is over or equal to 80<br/>
**Mean_cloud:** mean cloud<br/>
**population:** population density<br/>
**Sunshine_hour:** mean number of hour of sunshine<br/>
**Wind_direction:** mean wind direction<br/>
**Wind_speed:** mean wind speed<br/>
**Air_health_quality:** mean daily air health quality<br/>

**

## System Requirement:

**
Python 3.6<br/>
BeautifulSoup<br/>
Pandas<br/>
Numpy<br/>
Matplotlib<br/>
Seaborn<br/>
Openpyxl<br/>
Sklearn<br/>
wxPython<br/>

**

## Function of Program:

**
**‘Forecast’ Button**: Forecast the status of weather in the August of next year<br/>
**‘Activate Auto-Forecast’ Button**: Periodically forecast the status of weather<br/>
**‘Prediction’ Button**: Predict the mean temperature based on other factors<br/>

**

## User Guide:

**
**Step 1**: Download & Install Python 3.6<br/>
**Step 2**: Go to Terminal & Download Python Library (py -3.6 –m pip install ____)<br/>
**Step 3**: Go to ‘Weather_Prediction’ folder & click GUI.cpython-36<br/>
**Step 4**: The forecast result will be stored in ‘prediction’ folder. The prediction statistics will be stored in ‘statistics’ folder<br/>

**

## Developer Tools:

**
**Programming Language**: Python<br/>
**IDE**: PyCharm<br/>
**GUI**: wxPython, wxFormBuilder<br/>
**Web Scraping**: BeautifulSoup, ParseHub<br/>
**Debugging & Testing**: Jupyter Notebook<br/>
**Data Format**: Microsoft Excel<br/>

```

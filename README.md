# Weather-Forecast-And-Prediction-by-Machine-Learning

![Weather Forecast Image](https://user-images.githubusercontent.com/37294801/83221305-09402d80-a1a8-11ea-83e4-79625d909a15.png)
![Image 1](https://user-images.githubusercontent.com/37294801/83221317-11986880-a1a8-11ea-914e-9747e9c71066.png)
![Image 2](https://user-images.githubusercontent.com/37294801/83221323-152bef80-a1a8-11ea-837b-492af14a426f.png)
![Image 3](https://user-images.githubusercontent.com/37294801/83221359-32f95480-a1a8-11ea-86a3-8afbf754a2af.png)
![Image 4](https://user-images.githubusercontent.com/37294801/83221420-5ae8b800-a1a8-11ea-9aa5-755cc495aa57.png)
![Image 5](https://user-images.githubusercontent.com/37294801/83221401-502e2300-a1a8-11ea-8639-19284354128f.png)

## Background

For the current situation, the Hong Kong Observatory conducts traditional weather forecasting using four common methods: climatology, analog, persistence and trends, and numerical weather prediction. However, these methods have limitations and do not utilize machine learning algorithms. This project aims to increase the accuracy of weather prediction by applying machine learning techniques to forecast weather conditions at least one month in advance.

## Objective (Brief)

There are two purposes of this project. The first purpose is to forecast the status of weather in August of a specific year using decision tree regression. The second purpose is to predict the temperature using different algorithms such as linear regression, random forest regression, and K-nearest neighbor regression. The predictions will be based on multiple factors, including population density and air health quality.

![Experiment Image](https://lh3.googleusercontent.com/43WkMUHGBC12Fap74eYDH-rsIg7BgmaeAev2f_xhoa1hg678kmiQbIEawUfKkjOjsrvpzhzUIvy9)

## Purpose (Detail)

To forecast the status of weather in August of the next year:
- ML Algorithm: Decision Tree Regression
- Status: Wet and Heat
- Output Value: Yes / No

To predict the temperature using Different Algorithms:
- ML Algorithms: Linear Regression, Random Forest Regression, K-Nearest Neighbor
- Output Value: Numerical

Algorithm Details:
- Decision Tree: Builds regression or classification models in the form of a tree structure.
- Linear Regression: Predicts a dependent variable value based on an independent variable.
- Random Forest Regression: Performs regression and classification tasks using multiple decision trees and bagging.
- K-Nearest Neighbor Regression

## Data Description:

The dataset includes the following variables:
- `mean_temp`: Mean air temperature
- `max_temp`: Mean daily maximum air temperature
- `min_temp`: Mean daily minimum air temperature
- `meanhum`: Mean relative humidity
- `meandew`: Mean dew point temperature
- `pressure`: Mean daily air pressure
- `heat`: True when mean air temperature is over or equal to 30
- `wet`: True when mean relative humidity is over or equal to 80
- `mean_cloud`: Mean cloud coverage
- `population`: Population density
- `sunshine_hour`: Mean number of hours of sunshine
- `wind_direction`: Mean wind direction
- `wind_speed`: Mean wind speed
- `air_health_quality`: Mean daily air health quality

## Data Source:

The data for this project was obtained from the Hong Kong Observatory and aqhi.gov.hk.

## System Requirement:

Python 3.6 or above

The following Python libraries are required:
- BeautifulSoup
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Openpyxl
- Sklearn
- wxPython

## Function of Program:

- **'Forecast' Button**: Forecasts the status of weather in August of the next year.
- **'Activate Auto-Forecast' Button**: Periodically forecasts the status of weather.
- **'Prediction' Button**: Predicts the mean temperature based on other factors.

## User Guide:

1. Download and install Python 3.6 or above.
2. Open the terminal and install the required Python libraries using the following command: `pip install [library name]`
3. Navigate to the 'Weather_Prediction' folder and run the GUI.cpython-36 file.
4. The forecast results will be stored in the 'prediction' folder, and the prediction statistics will be stored in the 'statistics' folder.

## How to Contribute:

Contributions are welcome! Here's how you can contribute to this project:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request describing your changes.

## Developer Tools:

- Programming Language: Python
- IDE: PyCharm
- GUI: wxPython, wxFormBuilder
- Web Scraping: BeautifulSoup, ParseHub
- Debugging & Testing: Jupyter Notebook
- Data Format: Microsoft Excel

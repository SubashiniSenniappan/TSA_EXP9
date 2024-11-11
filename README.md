# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
# Developed by Subashini S
# Reg no:212222240106
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Step 1: Explore the dataset of weather
file_path = '/content/weather_classification_data.csv'
weather_data = pd.read_csv(file_path)

# Adding a simulated date range as the index if there isn't a time index
weather_data['Date'] = pd.date_range(start='2020-01-01', periods=len(weather_data), freq='D')
weather_data.set_index('Date', inplace=True)

# Select the column to forecast (e.g., Temperature)
data = weather_data['Temperature'].dropna()

# Step 2: Check for stationarity of time series
# Time series plot
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.title('Temperature Time Series')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

# ADF Test for stationarity
adf_result = adfuller(data)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
if adf_result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is not stationary. Differencing may be required.")

# Step 3: ACF and PACF Plots
plot_acf(data, lags=20)
plt.show()

plot_pacf(data, lags=20)
plt.show()

# Step 4: Transform to stationary (Differencing if needed)
# Differencing to make data stationary if not already stationary
data_diff = data.diff().dropna()

# Check stationarity after differencing (if necessary)
adf_result_diff = adfuller(data_diff)
print("ADF Statistic after differencing:", adf_result_diff[0])
print("p-value after differencing:", adf_result_diff[1])
if adf_result_diff[1] < 0.05:
    print("The differenced time series is stationary.")
else:
    print("The differenced time series is still not stationary. Further transformations may be needed.")

# Step 5: Determine ARIMA model parameters (p, q) using ACF and PACF plots
# This can be observed from the ACF and PACF plots or set manually. For simplicity, we assume p = 1, q = 1.

# Step 6: Fit the ARIMA model
# Using ARIMA(p=1, d=1, q=1) based on differencing result
p, d, q = 1, 1, 1
arima_model = ARIMA(data, order=(p, d, q))
arima_result = arima_model.fit()

# Step 7: Make time series predictions
forecast_steps = 30  # Forecasting the next 30 days
forecast = arima_result.forecast(steps=forecast_steps)

# Plotting historical data and forecast
plt.figure(figsize=(10, 4))
plt.plot(data, label='Historical Data')
plt.plot(pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'),
         forecast, label='Forecast', color='orange')
plt.title('Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Step 8: Evaluate model predictions
# Splitting data into train and test for evaluation
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fitting ARIMA model on train data for evaluation
model_validation = ARIMA(train, order=(p, d, q))
model_validation_fit = model_validation.fit()

# Forecasting on test data
forecast_test = model_validation_fit.forecast(steps=len(test))
mse = mean_squared_error(test, forecast_test)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/ed9a1ecf-f437-420b-aa14-65b79f1fb60c)


![Screenshot 2024-11-11 224838](https://github.com/user-attachments/assets/ea46c087-c8c6-453a-936f-24cf3fb4c269)
![image](https://github.com/user-attachments/assets/40f66b81-4e1c-4116-9934-c7b0bf924f0b)
![image](https://github.com/user-attachments/assets/b4614aa8-2e12-4b28-8a7f-45f254e46fb1)

![Screenshot 2024-11-11 225021](https://github.com/user-attachments/assets/875a6817-cd62-44c8-8fac-870d6a5429bd)
![image](https://github.com/user-attachments/assets/de84ff30-d204-4cb9-85f1-b49b8a9731db)

![Screenshot 2024-11-11 225137](https://github.com/user-attachments/assets/34d671ac-100c-4991-bacc-66799e6bed11)




### RESULT:
Thus the program run successfully based on the ARIMA model using python.

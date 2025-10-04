# Ex.No-6-TSA

## Developed By: Priyadharshan S
## Register Number: 212223240127

## Aim:
To implement the Holt Winters Method Model using Python

## Algorithm:
1.Import the necessary libraries

2.Load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration.

3.Resample it to a monthly frequency beginning of the month.

4.You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality

5.Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data

6.Create the final model and predict future data and plot it.

## Program:

```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("/content/co2_gr_mlo.csv", comment="#")

data['year'] = pd.to_datetime(data['year'], format='%Y')
data.set_index('year', inplace=True)

data.plot(title='Original Data')
plt.xlabel('Year')
plt.ylabel('Emissions')
plt.show()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[['ann inc']])
scaled_data = pd.Series(scaled.flatten(), index=data.index)

scaled_data.plot(title='Scaled Emission Data')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data['ann inc'], model="additive", period=11)


decomposition.plot()
plt.show()

scaled_data = scaled_data + 1 
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=11).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax = train_data.plot(label='Train Data')
test_data.plot(ax=ax, label='Test Data')
test_predictions_add.plot(ax=ax, label='Predictions')
ax.set_title('Train and Test Predictions')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Sunspots')
ax.legend()
plt.show()

import numpy as np
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'Test RMSE: {rmse:.4f}')

print(f'Scaled data mean: {scaled_data.mean():.4f}, sqrt(variance): {np.sqrt(scaled_data.var()):.4f}')

final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=11).fit()
future_steps = 10
final_predictions = final_model.forecast(steps=future_steps)
plt.figure(figsize=(14, 6))
ax = scaled_data.plot(label='Original Data')
final_predictions.plot(ax=ax, label='Forecast')
ax.set_title('Emission Forecast')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Emission')
ax.legend()
plt.show()

```
## Output:

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/9852d7c0-a1b4-48e9-ba42-5bcf144569d5" />
<img width="547" height="455" alt="image" src="https://github.com/user-attachments/assets/7b3f211c-fcee-4bf6-a976-f0f415e18399" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/76e9c4fe-41c2-42b4-9c00-592e86e110e6" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/0cfc103d-4113-462f-90bb-ad656526568b" />
<img width="1156" height="547" alt="image" src="https://github.com/user-attachments/assets/6c459f2d-f4a7-4817-adec-a960e2b648ce" />

## Result:
Thus, the program run successfully based on the Holt Winters Method model.


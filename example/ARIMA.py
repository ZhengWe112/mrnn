import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from lib.metrics import masked_mape_np
from lib.integrated_functions import growth_rate


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


filename = "../data/stock_data_y2016_g3_d_1_4_48_t64_p1.npz"
file_data = np.load(filename)

train_x1 = file_data['train_x1']  # (B, N, F)
train_x2 = file_data['train_x2']
train_x3 = file_data['train_x3']
train_target = file_data['train_target']

val_x1 = file_data['val_x1']
val_x2 = file_data['val_x2']
val_x3 = file_data['val_x3']
val_target = file_data['val_target']

test_x1 = file_data['test_x1']
test_x2 = file_data['test_x2']
test_x3 = file_data['test_x3']
test_target = file_data['test_target']

_max = file_data['mean']
_min = file_data['std']

test_x1 = max_min_normalization(test_x1, _max[0, 0, 0], _min[0, 0, 0])
test_target = max_min_normalization(test_target, _max[0, 0, 0], _min[0, 0, 0])

test_target = test_target[0:8000, :, :]

print(val_x1.shape)
prediction = []

for s in range(8000):
    print(s)
    data_flat = test_x1[s].flatten()
    model = ARIMA(data_flat, order=(3, 3, 1))
    model_fit = model.fit()

    n_forecast = 1
    forecast = model_fit.forecast(steps=n_forecast)

    prediction.append(forecast)  # (48)

prediction = np.concatenate(prediction, 0)  # (S, 48)

prediction = re_max_min_normalization(prediction, _max[0, 0, 0], _min[0, 0, 0])
data_target_tensor = re_max_min_normalization(test_target, _max[0, 0, 0], _min[0, 0, 0])

mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
corr = np.corrcoef(data_target_tensor.reshape(-1), prediction.reshape(-1))[0, 1]
print('all MAE: %.4f' % mae)
print('all RMSE: %.4f' % rmse)
print('all MAPE: %.4f' % mape)
print('all CORR: %.4f' % corr)

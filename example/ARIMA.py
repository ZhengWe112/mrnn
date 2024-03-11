import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def min_max_normalization(x1, x2, x3, target):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :param target: np.ndarray
    :return:
        stats: dict, two keys: mean and std
        tensor_norm: np.ndarray
    """
    _max = x1.max(axis=(0, 1), keepdims=True)
    _min = x1.min(axis=(0, 1), keepdims=True)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    x1_norm = normalize(x1)
    x2_norm = normalize(x2)
    x3_norm = normalize(x3)
    target_norm = normalize(target)
    return x1_norm, x2_norm, x3_norm, target_norm


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

train_x1, train_x2, train_x3, train_target = min_max_normalization(train_x1, train_x2, train_x3, train_target)
val_x1, val_x2, val_x3, val_target = min_max_normalization(val_x1, val_x2, val_x3, val_target)
test_x1, test_x2, test_x3, test_target = min_max_normalization(test_x1, test_x2, test_x3, test_target)

print(val_x1.shape)
tmp = []

for s in range(val_x1.shape[0]):
    print(s)
    data_flat = val_x1[s].flatten()
    target_flat = val_target[s].flatten()
    model = ARIMA(data_flat, order=(3, 3, 1))
    model_fit = model.fit()

    n_forecast = 1
    forecast = model_fit.forecast(steps=n_forecast)

    loss = mean_squared_error(target_flat, forecast)
    tmp.append(loss)


print(sum(tmp) / len(tmp))
# stock_data_y2016_g3_d_1_4_48          0.35200548491236533
# stock_data_y2016_g3_d_1_4_48_t64_p1   0.0011552321697535755

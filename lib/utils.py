import numpy as np
import torch
import torch.utils.data
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .metrics import masked_mape_np
from time import time


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def frobenius_norm(tensor):
    norms = torch.norm(tensor, 'fro', dim=0)
    return norms


def load_stock_data(filename, DEVICE, batchsz):
    """
    loading data and converting to tensor
    :param filename: str
    :param DEVICE: torch.device
    :param batchsz: int
    :return:
        three DataLoaders, each dataloader contains:
        train_x_tensor: (B, T, F)
        train_target_tensor: (B, T, F)
    """
    print("load file: ", filename)
    file_data = np.load(filename)

    _max = file_data['mean']
    _min = file_data['std']

    train_x1 = max_min_normalization(file_data['train_x1'], _max, _min)  # (B, T, F)
    train_x2 = max_min_normalization(file_data['train_x2'], _max, _min)
    train_x3 = max_min_normalization(file_data['train_x3'], _max, _min)
    train_target = max_min_normalization(file_data['train_target'], _max, _min)

    val_x1 = max_min_normalization(file_data['val_x1'], _max, _min)
    val_x2 = max_min_normalization(file_data['val_x2'], _max, _min)
    val_x3 = max_min_normalization(file_data['val_x3'], _max, _min)
    val_target = max_min_normalization(file_data['val_target'], _max, _min)

    test_x1 = max_min_normalization(file_data['test_x1'], _max, _min)
    test_x2 = max_min_normalization(file_data['test_x2'], _max, _min)
    test_x3 = max_min_normalization(file_data['test_x3'], _max, _min)
    test_target = max_min_normalization(file_data['test_target'], _max, _min)

    train_x1_tensor = torch.from_numpy(train_x1).type(torch.FloatTensor).to(DEVICE)
    train_x2_tensor = torch.from_numpy(train_x2).type(torch.FloatTensor).to(DEVICE)
    train_x3_tensor = torch.from_numpy(train_x3).type(torch.FloatTensor).to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)
    train_datasets = torch.utils.data.TensorDataset(train_x1_tensor, train_x2_tensor,
                                                    train_x3_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batchsz, shuffle=True)

    val_x1_tensor = torch.from_numpy(val_x1).type(torch.FloatTensor).to(DEVICE)
    val_x2_tensor = torch.from_numpy(val_x2).type(torch.FloatTensor).to(DEVICE)
    val_x3_tensor = torch.from_numpy(val_x3).type(torch.FloatTensor).to(DEVICE)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)
    val_datasets = torch.utils.data.TensorDataset(val_x1_tensor, val_x2_tensor,
                                                  val_x3_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batchsz)

    test_x1_tensor = torch.from_numpy(test_x1).type(torch.FloatTensor).to(DEVICE)
    test_x2_tensor = torch.from_numpy(test_x2).type(torch.FloatTensor).to(DEVICE)
    test_x3_tensor = torch.from_numpy(test_x3).type(torch.FloatTensor).to(DEVICE)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)
    test_datasets = torch.utils.data.TensorDataset(test_x1_tensor, test_x2_tensor,
                                                   test_x3_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batchsz)

    print('train:', train_x1_tensor.size(), train_x2_tensor.size(), train_x3_tensor.size(), train_target_tensor.size())
    print('val:', val_x1_tensor.size(), val_x2_tensor.size(), val_x3_tensor.size(), val_target_tensor.size())
    print('test:', test_x1_tensor.size(), test_x2_tensor.size(), test_x3_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min


def predict_and_save_results(model, data_loader, data_target_tensor, epoch, params_path, _max, _min, type):
    """
    predict and save results
    :param model: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param params_path: string, the path for saving the results
    :param _max: (1, 1, 1)
    :param _min: (1, 1, 1)
    :param type: string
    :return:
    """
    model.train(False)  # ensure dropout layers are in test mode
    start_time = time()

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        prediction = []  # prediction value

        for batchIdx, (x1, x2, x3, labels) in enumerate(data_loader):
            x = [x1, x2, x3]
            out = model(x)
            prediction.append(out.detach().cpu().numpy())

        print('test time on whole data:%.2fs' % (time() - start_time))

        prediction = np.concatenate(prediction, 0)  # (S, T, 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0], _min[0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0], _min[0, 0, 0])

        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[1]

        # for i in range(prediction_length):
        #     assert data_target_tensor.shape[0] == prediction.shape[0]
        #     mae = mean_absolute_error(data_target_tensor[:, i], prediction[:, i])
        #     rmse = mean_squared_error(data_target_tensor[:, i], prediction[:, i]) ** 0.5
        #     mape = masked_mape_np(data_target_tensor[:, i], prediction[:, i], 0)
        #     print('MAE: %.6f' % mae)
        #     print('RMSE: %.6f' % rmse)
        #     print('MAPE: %.6f' % mape)
        #     excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        corr = np.corrcoef(data_target_tensor.reshape(-1), prediction.reshape(-1))[0, 1]
        print('all MAE: %.2f' % mae)
        print('all RMSE: %.2f' % rmse)
        print('all MAPE: %.2f' % mape)
        print('all CORR: %.2f' % corr)
        excel_list.extend([mae, rmse, mape])
        print(excel_list)


def predict_and_save_results_without_granularity(model, data_loader, data_target_tensor, epoch, params_path, _max, _min, type):
    """
    predict and save results
    :param model: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param params_path: string, the path for saving the results
    :param _max: (1, 1, 1)
    :param _min: (1, 1, 1)
    :param type: string
    :return:
    """
    model.train(False)  # ensure dropout layers are in test mode
    start_time = time()

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        prediction = []  # prediction value

        for batchIdx, (x, _, __, labels) in enumerate(data_loader):
            xt, yt = model(x)
            prediction.append(yt.detach().cpu().numpy())

        print('test time on whole data:%.2fs' % (time() - start_time))

        prediction = np.concatenate(prediction, 0)  # (S, T, 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0], _min[0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0], _min[0, 0, 0])

        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[1]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]

            mae = mean_absolute_error(data_target_tensor[:, i], prediction[:, i])
            rmse = mean_squared_error(data_target_tensor[:, i], prediction[:, i]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, i], prediction[:, i], 0)
            print('MAE: %.6f' % mae)
            print('RMSE: %.6f' % rmse)
            print('MAPE: %.6f' % mape)
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % mae)
        print('all RMSE: %.2f' % rmse)
        print('all MAPE: %.2f' % mape)
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

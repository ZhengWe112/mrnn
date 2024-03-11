import pandas as pd
import numpy as np
import os
import argparse
import configparser


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/SD2016.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config, flush=True)
config.read(args.config)

data_config = config['Data']
train_config = config['Training']

num_for_predict = int(data_config['num_for_predict'])
num_for_train = int(data_config['num_for_train'])
daydata_path = data_config['daydata_path']  # 股票天数据位置
minutedata_path = data_config['minutedata_path']  # 股票分钟数据位置
start_date = data_config['start_date']
end_data = data_config['end_data']
year = data_config['year']
intervals = list(map(int, train_config['intervals'].split(',')))
granularity = train_config['granularity']
coarsest_data = train_config['coarsest_data']


# 导入某文件夹下所有股票的代码
def get_stock_code_list_in_one_dir(path):
    """
    从指定文件夹下，导入所有csv文件的文件名
    :param path:
    :return:
    """
    stock_list = []

    # 系统自带函数os.walk，用于遍历文件夹中的所有文件
    for root, dirs, files in os.walk(path):
        if files:  # 当files不为空的时候
            for f in files:
                if f.endswith('.csv'):
                    stock_list.append(f[:8])

    return sorted(stock_list)


def get_return_rate(day_seq, code):
    """
    calculate return_data
    :param day_seq:
    :param code: the code of stock, aiming to get minute data
    :return: return_rate
    """
    day_seq['收益率'] = (day_seq['收盘价'] - day_seq['前收盘价']) / day_seq['前收盘价']
    day_seq['交易日期'] = pd.to_datetime(day_seq['交易日期'])
    day_seq.set_index('交易日期', inplace=True)
    day_seq = day_seq.loc[start_date: end_data]
    day_return_rate = day_seq['收益率'].values
    day_return_rate = day_return_rate[:, None]  # (seq_len, feature)

    subroot = os.listdir(minutedata_path)
    all_hour_return_rate = np.empty((0, 1))  # (seq_len, feature)
    all_min5_return_rate = np.empty((0, 1))  # (seq_len, feature)

    code = code[2:8] + '.' + code[:2].upper() + '1'

    for sub in subroot:
        fp = os.path.join(minutedata_path, sub)
        for name in os.listdir(fp):
            filename = os.path.join(fp, name)
            minute_data = pd.read_csv(filename, encoding='gbk', parse_dates=['Date'])
            condition = minute_data['Code'] == code
            minute_data = minute_data[condition]

            hour_data = minute_data.iloc[0:len(minute_data):60]
            hour_data['小时收益率'] = hour_data['Close'].pct_change()
            hour_data = hour_data.iloc[1:]
            hour_return_rate = hour_data['小时收益率'].values
            hour_return_rate = hour_return_rate[:, None]
            all_hour_return_rate = np.concatenate((hour_return_rate, all_hour_return_rate), axis=0)

            min5_data = minute_data.iloc[0:len(minute_data):5]
            min5_data['5分钟收益率'] = min5_data['Close'].pct_change()
            min5_data = min5_data.iloc[1:]
            min5_return_rate = min5_data['5分钟收益率'].values
            min5_return_rate = min5_return_rate[:, None]
            all_min5_return_rate = np.concatenate((min5_return_rate, all_min5_return_rate), axis=0)

    return day_return_rate, all_hour_return_rate, all_min5_return_rate


def get_sample_indices(day_return_rate, hour_return_rate, min5_return_rate,
                       predict_start_idx, num_for_predict, num_for_train=12):
    """
    :param day_return_rate: np.ndarray, shape is (sequence_length, num_of_features)
    :param hour_return_rate: np.ndarray, shape is (sequence_length * 4, num_of_features)
    :param min5_return_rate: np.ndarray, shape is (sequence_length * 4 * 12, num_of_features)
    :param predict_start_idx: int, the first index of predicting target
    :param num_for_predict: int, the number of points will be predicted for each sample
    :param num_for_train: int, the number of points will be used to train for each sample
    :return:
        x1, x2, x3: np.ndarray, three granularity data
        target: np.ndarray
    """
    x1, x2, x3 = None, None, None

    if num_for_train <= 0:
        raise ValueError("num_for_train should be greater than 0!")

    if predict_start_idx + num_for_predict > day_return_rate.shape[0]:
        return x1, x2, x3, None

    if predict_start_idx - num_for_train < 0:
        return x1, x2, x3, None

    start_idx = predict_start_idx - num_for_train
    end_idx = predict_start_idx
    x1 = day_return_rate[start_idx: end_idx]
    x2 = hour_return_rate[start_idx*4: end_idx*4]
    x3 = min5_return_rate[start_idx*4*12: end_idx*4*12]
    target = day_return_rate[predict_start_idx: predict_start_idx + num_for_predict]
    return x1, x2, x3, target


def read_and_generate_dataset_on_single_stock(day_file, code, num_for_predict, num_for_train):
    """
    :param day_file: str, path of daily data csv file
    :param code: str, stock code
    :param num_for_predict: int
    :param num_for_train: int
    :return:
    """
    day_seq = pd.read_csv(day_file, encoding='gbk', skiprows=1, parse_dates=['交易日期'])
    day_return_rate, hour_return_rate, min5_return_rate = get_return_rate(day_seq, code)

    stock_samples = []

    for idx in range(day_return_rate.shape[0]):
        sample = get_sample_indices(day_return_rate, hour_return_rate, min5_return_rate,
                                    idx, num_for_predict, num_for_train)
        if sample[0] is None:
            continue

        x1, x2, x3, target = sample
        sample = []
        x1 = np.expand_dims(x1, axis=0)  # (1, seq_len, feature)
        x2 = np.expand_dims(x2, axis=0)  # (1, seq_len, feature)
        x3 = np.expand_dims(x3, axis=0)  # (1, seq_len, feature)
        target = np.expand_dims(target, axis=0)  # (1, seq_len, feature)
        sample.append(x1)
        sample.append(x2)
        sample.append(x3)
        sample.append(target)

        stock_samples.append(sample)

    return stock_samples


def read_and_generate_dataset(daydata_path, num_for_predict, num_for_train, save=True):
    stock_code_list = get_stock_code_list_in_one_dir(daydata_path)

    all_samples = []

    for code in stock_code_list[500:1000]:
        print(code)
        day_file = daydata_path + '/%s.csv' % code
        stock_sample = read_and_generate_dataset_on_single_stock(day_file, code, num_for_predict, num_for_train)
        for sample in stock_sample:
            all_samples.append(sample)

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    all_data = {
        'train': {
            'x1': training_set[0],
            'x2': training_set[1],
            'x3': training_set[2],
            'target': training_set[3],
        },
        'val': {
            'x1': validation_set[0],
            'x2': validation_set[1],
            'x3': validation_set[2],
            'target': validation_set[3],
        },
        'test': {
            'x1': testing_set[0],
            'x2': testing_set[1],
            'x3': testing_set[2],
            'target': testing_set[3],
        },
    }
    print('train x1:', all_data['train']['x1'].shape)
    print('train x2:', all_data['train']['x2'].shape)
    print('train x3:', all_data['train']['x3'].shape)
    print('train target:', all_data['train']['target'].shape)
    print()
    print('val x1:', all_data['val']['x1'].shape)
    print('val x2:', all_data['val']['x2'].shape)
    print('val x3:', all_data['val']['x3'].shape)
    print('val target:', all_data['val']['target'].shape)
    print()
    print('test x1:', all_data['test']['x1'].shape)
    print('test x2:', all_data['test']['x2'].shape)
    print('test x3:', all_data['test']['x3'].shape)
    print('test target:', all_data['test']['target'].shape)
    print()

    if save:
        stock_data_filename = 'stock_data_y%s_g%s_%s_p1' % (year, granularity, coarsest_data)
        for i in intervals:
            stock_data_filename += '_%s' % i

        fp = open(args.config, 'w')
        config.set('Data', 'stock_data_filename', 'data/%s' % stock_data_filename)
        config.write(fp)
        dirpath = os.path.dirname('./data/')
        new_filename = os.path.join(dirpath, stock_data_filename)
        print('save file:', new_filename)
        np.savez_compressed(new_filename,
                            train_x1=all_data['train']['x1'], train_x2=all_data['train']['x2'],
                            train_x3=all_data['train']['x3'], train_target=all_data['train']['target'],
                            val_x1=all_data['val']['x1'], val_x2=all_data['val']['x2'],
                            val_x3=all_data['val']['x3'], val_target=all_data['val']['target'],
                            test_x1=all_data['test']['x1'], test_x2=all_data['test']['x2'],
                            test_x3=all_data['test']['x3'], test_target=all_data['test']['target'],
                            )
    return all_data


all_data = read_and_generate_dataset(daydata_path, num_for_predict, num_for_train)

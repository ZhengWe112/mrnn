import pandas as pd
import numpy as np
import os
import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/LD2012_2014.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config, flush=True)
config.read(args.config)

data_config = config['Data']
train_config = config['Training']

num_for_predict = int(data_config['num_for_predict'])
num_for_train = int(data_config['num_for_train'])
start_date = data_config['start_date']
end_data = data_config['end_data']
intervals = list(map(int, train_config['intervals'].split(',')))
granularity = train_config['granularity']
coarsest_data = train_config['coarsest_data']
data_file = data_config['data_filename']

df = pd.read_csv(data_file, sep=';', low_memory=False)

df.rename(columns={df.columns[0]: 'date'}, inplace=True)
df.set_index('date', inplace=True)
df = df.loc[start_date: end_data]

df = df.values


def convert_to_float(s):
    if ',' in s:
        integer_part, decimal_part = s.split(',')

        integer_part = int(integer_part)
        decimal_part = int(decimal_part)

        decimal_value = decimal_part / (10 ** len(str(decimal_part)))

        return float(integer_part + decimal_value)
    else:

        return float(s)


def cal_coarse(seq, interval):
    res = np.empty((0, 1))
    for i in range(0, len(seq), interval):
        tmp = seq[i:i + interval, :]
        tmp = np.sum(tmp, axis=0, keepdims=True)
        res = np.concatenate((res, tmp), axis=0)

    return res


def cal_mean_and_std(train):
    """
    Parameters
    ----------
    train: np.ndarray (B, T, F)
    Returns
    ----------
    stats: dict, two keys: mean and std
    """
    _max = train.max(axis=(0, 1), keepdims=True)
    _min = train.min(axis=(0, 1), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    return {'_max': _max, '_min': _min}


def get_sample_indices(seq1, seq2, seq3, seq4, seq5, predict_start_idx, num_for_predict, num_for_train, intervals):
    """
    :param seq1, .., seq5: np.ndarray, a customer's data from 2012 to 2014
    :param predict_start_idx: int, the first index of predicting target
    :param num_for_predict: int, the number of points will be predicted for each sample
    :param num_for_train: int, the number of points will be used to train for each sample
    :param intervals: list, Assume there are five granularities
    :return:
        x1, x2, x3, x4, x5: np.ndarray, five granularity data
        target: np.ndarray
    """

    x1, x2, x3, x4, x5 = None, None, None, None, None

    if predict_start_idx + num_for_predict > seq1.shape[0]:
        return x1, x2, x3, x4, x5, None

    if predict_start_idx - num_for_train < 0:
        return x1, x2, x3, x4, x5, None

    start_idx = predict_start_idx - num_for_train
    end_idx = predict_start_idx
    x1 = seq1[start_idx: end_idx]
    x2 = seq2[start_idx*2: end_idx*2]
    x3 = seq3[start_idx*6: end_idx*6]
    x4 = seq4[start_idx*24: end_idx*24]
    x5 = seq5[start_idx*96: end_idx*96]
    target = seq1[predict_start_idx: predict_start_idx + num_for_predict]
    return x1, x2, x3, x4, x5, target


def read_and_generate_dataset(all_data, num_for_predict, num_for_train, intervals, save=True):

    all_samples = []

    for cust in range(all_data.shape[1]):  # all_data.shape[1]
        print(cust)
        seq = all_data[:, cust:cust+1]
        seq1 = cal_coarse(seq, intervals[4] // intervals[0])
        seq2 = cal_coarse(seq, intervals[4] // intervals[1])
        seq3 = cal_coarse(seq, intervals[4] // intervals[2])
        seq4 = cal_coarse(seq, intervals[4] // intervals[3])
        seq5 = seq
        print(seq.shape)
        for idx in range(0, seq.shape[0], num_for_train):
            sample = get_sample_indices(seq1, seq2, seq3, seq4, seq5, idx, num_for_predict, num_for_train, intervals)
            if sample[0] is None:
                continue

            x1, x2, x3, x4, x5, target = sample
            sample = []
            x1 = np.expand_dims(x1, axis=0)  # (1, seq_len, feature)
            x2 = np.expand_dims(x2, axis=0)  # (1, seq_len, feature)
            x3 = np.expand_dims(x3, axis=0)  # (1, seq_len, feature)
            x4 = np.expand_dims(x4, axis=0)  # (1, seq_len, feature)
            x5 = np.expand_dims(x5, axis=0)  # (1, seq_len, feature)
            target = np.expand_dims(target, axis=0)  # (1, seq_len, feature)
            sample.append(x1)
            sample.append(x2)
            sample.append(x3)
            sample.append(x4)
            sample.append(x5)
            sample.append(target)

            all_samples.append(sample)

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    _max, _min = [], []
    for xi in training_set:
        _max.append(cal_mean_and_std(xi)['_max'])
        _min.append(cal_mean_and_std(xi)['_min'])
    _max = np.concatenate([i for i in _max], axis=2)  # (1, 1, nb_granularity)
    _min = np.concatenate([i for i in _min], axis=2)  # (1, 1, nb_granularity)

    all_data = {
        'train': {
            'x1': training_set[0],
            'x2': training_set[1],
            'x3': training_set[2],
            'x4': training_set[3],
            'x5': training_set[4],
            'target': training_set[5],
        },
        'val': {
            'x1': validation_set[0],
            'x2': validation_set[1],
            'x3': validation_set[2],
            'x4': validation_set[3],
            'x5': validation_set[4],
            'target': validation_set[5],
        },
        'test': {
            'x1': testing_set[0],
            'x2': testing_set[1],
            'x3': testing_set[2],
            'x4': testing_set[3],
            'x5': testing_set[4],
            'target': testing_set[5],
        },
        'stats': {
            '_max': _max,
            '_min': _min,
        }
    }
    print('train x1:', all_data['train']['x1'].shape)
    print('train x2:', all_data['train']['x2'].shape)
    print('train x3:', all_data['train']['x3'].shape)
    print('train x4:', all_data['train']['x4'].shape)
    print('train x5:', all_data['train']['x5'].shape)
    print('train target:', all_data['train']['target'].shape)
    print()
    print('val x1:', all_data['val']['x1'].shape)
    print('val x2:', all_data['val']['x2'].shape)
    print('val x3:', all_data['val']['x3'].shape)
    print('val x4:', all_data['val']['x4'].shape)
    print('val x5:', all_data['val']['x5'].shape)
    print('val target:', all_data['val']['target'].shape)
    print()
    print('test x1:', all_data['test']['x1'].shape)
    print('test x2:', all_data['test']['x2'].shape)
    print('test x3:', all_data['test']['x3'].shape)
    print('test x4:', all_data['test']['x4'].shape)
    print('test x5:', all_data['test']['x5'].shape)
    print('test target:', all_data['test']['target'].shape)
    print()

    if save:
        stock_data_filename = 'LD_data_%s_t16_p1' % coarsest_data
        dirpath = os.path.dirname('./data/')
        new_filename = os.path.join(dirpath, stock_data_filename)
        print('save file:', new_filename)
        np.savez_compressed(new_filename,
                            train_x1=all_data['train']['x1'], train_x2=all_data['train']['x2'],
                            train_x3=all_data['train']['x3'], train_x4=all_data['train']['x4'],
                            train_x5=all_data['train']['x5'], train_target=all_data['train']['target'],
                            val_x1=all_data['val']['x1'], val_x2=all_data['val']['x2'],
                            val_x3=all_data['val']['x3'], val_x4=all_data['val']['x4'],
                            val_x5=all_data['val']['x5'], val_target=all_data['val']['target'],
                            test_x1=all_data['test']['x1'], test_x2=all_data['test']['x2'],
                            test_x3=all_data['test']['x3'], test_x4=all_data['test']['x4'],
                            test_x5=all_data['test']['x5'], test_target=all_data['test']['target'],
                            mean=all_data['stats']['_max'], std=all_data['stats']['_min']
                            )
    return all_data


for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        df[i, j] = convert_to_float(str(df[i, j]))

read_and_generate_dataset(df, num_for_predict, num_for_train, intervals)

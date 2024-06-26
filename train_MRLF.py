import torch
import numpy as np
import os
import configparser
import argparse
import shutil
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn import functional as F
from model.mrlf import mrlf
from model.CustomLoss import CustomLoss
from lib.utils import load_stock_data, predict_and_save_results

# python -u train_MRLF.py --config configurations/SD2016.conf --cuda=0 > model.out &

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/SD2016.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)

config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config, flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
stock_data_filename = data_config['stock_data_filename']
batch_size = int(training_config['batch_size'])
feature_size = int(training_config['feature_size'])
hidden_size = int(training_config['hidden_size'])
model_name = training_config['model_name']
num_for_predict = int(data_config['num_for_predict'])
num_for_train = int(data_config['num_for_train'])
intervals = list(map(int, training_config['intervals'].split(',')))

params_path = './experiments/%s_%s' % (model_name, stock_data_filename.split('/')[1].split('.')[0])

(train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader,
 test_target_tensor, _max, _min) = load_stock_data(stock_data_filename, DEVICE, batch_size)

model = mrlf(feature_size, hidden_size, num_for_train, num_for_predict, intervals).to(DEVICE)
print(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
# criterion = CustomLoss(DEVICE)
criterion = nn.L1Loss().to(DEVICE)


def train_main():
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
    else:
        os.makedirs(params_path)

    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    global_step = 0
    best_val_loss = np.inf
    best_epoch = 0

    for epoch in range(0, epochs):
        # train
        model.train()
        print(f'epoch: {epoch}')
        for batchIdx, (x1, x2, x3, label) in enumerate(train_loader):
            x1, x2, x3, label = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), label.to(DEVICE)
            x = [x1, x2, x3]
            out, recloss = model(x)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sw.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        model.eval()
        with torch.no_grad():
            tmp = []
            for x1, x2, x3, label in val_loader:
                x1, x2, x3, label = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), label.to(DEVICE)
                x = [x1, x2, x3]
                out, recloss = model(x)
                loss = F.mse_loss(out, label)
                tmp.append(loss)

            val_loss = sum(tmp) / len(tmp)
            sw.add_scalar('validation_loss', val_loss, epoch)

            if val_loss < best_val_loss:
                params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), params_filename)
                print('save parameters to file: %s' % params_filename, flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test')


def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type):
    """
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 1), used in renormalization
    :param _min: (1, 1, 1)
    :param type: string
    :return:
    """

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
    print('load weight from:', params_filename, flush=True)

    model.load_state_dict(torch.load(params_filename))
    predict_and_save_results(model, data_loader, data_target_tensor, epoch, params_path, _max, _min, type)


if __name__ == '__main__':
    train_main()

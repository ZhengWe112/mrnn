# MRNN
tensorboard --logdir ./experiments

import numpy as np

data = np.load('./data/stock_data_y2016_g3_d_1_4_48_t64_p1.npz')
print(data.files)
a = data['train_x1']

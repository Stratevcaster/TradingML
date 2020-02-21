'''
Created on Feb 14, 2020

@author: USER
'''

import os
import time
from tensorflow.keras.layers import LSTM, GRU,RNN
from keras import backend as K
import tensorflow as tf
# TAMAÑO DE LA VENTANA O SECUENCIA
N_STEPS = 70
#  SIGUIENTE DIA
LOOKUP_STEP = 8

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")
fecha_modelos="2020-02-20"

### model parameters

N_LAYERS = 4
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4

### training parameters

# mean squared error loss
LOSS = "mse"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 12

# Apple stock market
ticker = "TSLA"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")



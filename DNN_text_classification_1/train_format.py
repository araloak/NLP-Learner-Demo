from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import pandas as pd
#写好model.py文件用以构造模型
from model import * 
from data_helper import load_data

np.set_printoptions(threshold=np.inf)
pd.set_option('precision', 7) #设置精度
pd.set_option('display.float_format', lambda x: '%.5f' % x) 

seed=2019
epochs=100
batch_size=128
valid_data_size=0.33
patience=10
epochs=150
model_path=sys.path[0]+"//model//"
task="binary"
np.random.seed(seed)
if __name__ == "__main__":
    #data为矩阵,每行代表一个样本,label为分类标签，可以是多分类或二分类，所有数据格式已经在data_helper文件处理好
    data,label=load_data()
    print('data.shape',data.shape)
    print('label.shape',label.shape)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=patience, mode='auto')
    checkpoint = ModelCheckpoint(model_path+'weights.{epoch:03d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto') #监控验证集准确率并保存每次训练最好模型
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1) #早停机制，验证集loss不再下降便停止
    
    model = create_model(data.shape[1],label.shape[1],task) #input_dim是输入层神经元个数，output_dim是输出层神经元个数
    
    train,valid,train_label,valid_label= train_test_split(data, label,test_size=valid_data_size,random_state=seed)
    model.fit(train, train_label, epochs=epochs, batch_size=batch_size, verbose=1,validation_data=(valid,valid_label),callbacks=[checkpoint,reduce_lr,early_stopping])

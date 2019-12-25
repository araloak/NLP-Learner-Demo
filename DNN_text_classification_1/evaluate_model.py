from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
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



fold=10  #k_折交叉验证折数
seed=2019
epochs=10
batch_size=128
patience=3
valid_data_size=0.33
patience=10
epochs=150
batch_size=10
model_path=sys.path[0]+"//model//"
task="multiclass"
np.random.seed(seed)
if __name__ == "__main__":
    #data为矩阵,每行代表一个样本,label为分类标签，可以是多分类或二分类，所有数据格式已经在data_helper文件处理好
    data,label=load_data()
    
    # define 10-fold cross validation test harness 十折交叉验证k一般取值为5或者10
    kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    cvscores = []
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=patience, mode='auto') 
    checkpoint = ModelCheckpoint(model_path+'weights.{epoch:03d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1) #早停机制，验证集loss不再下降便停止
    count=0
    for train, test in kfold.split(data, label[:,0]): #StratifiedKFold要求输入的label必须是一维向量
        count+=1
        print('这是第'+str(count)+'折\n')
        model = create_model(data.shape[1],label.shape[1],task) #input_dim是输入层神经元个数，output_dim是输出层神经元个数
        train_data,valid_data,train_label,valid_label= train_test_split(data[train], label[train],test_size=valid_data_size,random_state=seed)
        model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, verbose=0,validation_data=(valid_data,valid_label),callbacks=[checkpoint,reduce_lr,early_stopping])
        # evaluate the model
        scores = model.evaluate(data[test], label[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("平均正确率和标准差%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

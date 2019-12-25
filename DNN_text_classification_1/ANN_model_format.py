from keras import Sequential
from keras.layers import Dense,Dropout
import numpy as np

#input_dim是输入数据维度，output_dim是输出层神经元个数
def creat_model(input_dim,output_dim,task):
    #在data_helper文件中定义好训练数据和标签的获取函数load_data()
    if task=="binary":
        activation="sigmoid"
        loss='binary_crossentropy'
    elif task=="multiclass":
        activation="softmax"
        loss='categorical_crossentropy'
    model = Sequential([
    Dense(200,input_dim=input_dim，activation='relu'),
    Dropout(0.2),
    Dense(200,activation='relu'),
    Dropout(0.2),
    Dense(output_dim,activation=activation)    
    ])
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model

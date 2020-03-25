import tensorflow as tf

from keras_bert import load_trained_model_from_checkpoint
from keras.layers import *
from keras.models import Model
from keras import backend as K
<<<<<<< HEAD
from keras_radam import RAdam
=======
from keras.optimizers import Adam
>>>>>>> 优化了代码

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_bert(args):
    bert_model = load_trained_model_from_checkpoint(args.config_path, args.checkpoint_path, seq_len=None)  #加载预训练模型
 
    for l in bert_model.layers:
<<<<<<< HEAD
        l.trainable = True
 
=======
            l.trainable = True
            
>>>>>>> 优化了代码
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    
    p = Dense(args.nclass, activation='softmax')(x)
 
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
<<<<<<< HEAD
                  optimizer=RAdam(args.lr),    #用足够小的学习率
=======
                  optimizer=Adam(args.lr),    #用足够小的学习率
>>>>>>> 优化了代码
                  metrics=['accuracy',f1])
    print(model.summary())
    return model
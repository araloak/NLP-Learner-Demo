import tensorflow as tf
import keras

from keras_bert import load_trained_model_from_checkpoint
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras_bert.optimizers import AdamWarmup

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
        #if "-12-" in l.name or "-11-" in l.name or "-10-" in l.name:
        l.trainable = True#选择冻结的层
 
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    
    p = Dense(args.nclass, activation='softmax')(x)
 
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',#多分类
                  optimizer=Adam(args.lr),    #用足够小的学习率
                  metrics=['accuracy',f1])
    print(model.summary())
    return model

def pretrain_bert(args,training= True):
    bert_model = load_trained_model_from_checkpoint(args.config_path, args.checkpoint_path, seq_len=None,training = training)  #加载预训练模型
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in,x3_in])

    model = Model([x1_in, x2_in,x3_in], x)
    
    model.compile(
        optimizer=AdamWarmup(
            decay_steps=100000,
            warmup_steps=10000,
            learning_rate=1e-4,
            weight_decay=0.01,
            weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],
        ),
        loss=keras.losses.sparse_categorical_crossentropy,
    )
    print(model.summary())
    return model

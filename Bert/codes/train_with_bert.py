import os
#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'#使用服务器添加的指令
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback
from get_bert_model import *
from data_helper import *
import argparse
<<<<<<< HEAD

=======
import numpy as np
from keras.callbacks import TensorBoard
import datetime
>>>>>>> 优化了代码

parser = argparse.ArgumentParser()
parser.add_argument('--times', default=1, type=int, required=False, help='第几次训练')
parser.add_argument('--nclass', default=3, type=int, required=False, help='几分类')
parser.add_argument('--epoch', default=2, type=int, required=False, help='训练批次')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='训练批次')
<<<<<<< HEAD
parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch_size')
parser.add_argument('--dev_size', default=0.2, type=float, required=False, help='test_size')
parser.add_argument('--maxlen', default=512, type=int, required=False, help='训练样本最大句子长度')
parser.add_argument('--pretrained_path', default="D:/codes/Xlnet/chinese_roberta_wwm_ext_L-12_H-768_A-12/", type=str, required=False, help='预训练模型保存目录')
parser.add_argument('--submision_sample_path', default="../subs/submit_example.csv", type=str, required=False, help='预测结果提交文件')
parser.add_argument('--do_train', default=True,action='store_true', required=False, help='是否训练')
parser.add_argument('--do_predict', default=False, action='store_true', required=False, help='提交测试')
=======
parser.add_argument('--batch_size', default=1, type=int, required=False, help='batch_size')
parser.add_argument('--dev_size', default=0.2, type=float, required=False, help='test_size')
parser.add_argument('--maxlen', default=512, type=int, required=False, help='训练样本最大句子长度')
parser.add_argument('--pretrained_path', default="D:/codes/Bert_projects/pre_trained_models/chinese_L-12_H-768_A-12/", type=str, required=False, help='预训练模型保存目录')
parser.add_argument('--submision_sample_path', default="../subs/submit_example.csv", type=str, required=False, help='预测结果提交文件')
parser.add_argument('--do_train', default=True,action='store_true', required=False, help='是否训练')
parser.add_argument('--do_predict', default=True, action='store_true', required=False, help='提交测试')
>>>>>>> 优化了代码

args = parser.parse_args()

args.train_path = "../data/"+str(args.times)+"/train.tsv"
args.test_path = "../data/"+str(args.times)+"/test.tsv"
args.sub_path = "../subs/"+str(args.times)+"/"
args.finetuned_path = "../models/"+str(args.times)+"/"

args.config_path = args.pretrained_path + 'bert_config.json'
<<<<<<< HEAD
args.model_path = args.pretrained_path + 'bert_model.ckpt'
=======
args.checkpoint_path = args.pretrained_path + 'bert_model.ckpt'
>>>>>>> 优化了代码
args.vocab_path = args.pretrained_path + 'vocab.txt'

if os.path.exists(args.sub_path)==False:
    os.mkdir(args.sub_path, mode=0o777)
if os.path.exists(args.finetuned_path)==False:
    os.mkdir(args.finetuned_path, mode=0o777)

class Metrics(Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict([self.validation_data[0],self.validation_data[1]]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return

def train():
    data,label = get_data(args.train_path) #data是由所有句子组成的一级列表["你是人","你不是人"]

    print(label[0:10],data[0:10])
    x_train,x_test, y_train, y_test =train_test_split(data,label,test_size=args.dev_size, random_state=2020)

    x1_train,x2_train=get_bert_input(x_train,args.vocab_path,args.maxlen)#x1、x2
    x1_test,x2_test=get_bert_input(x_test,args.vocab_path,args.maxlen)#x1、x2

    model=build_bert(args)

    checkpoint = ModelCheckpoint('../models/'+str(args.times)+'/weights.{epoch:03d}-{val_f1:.4f}.h5', monitor='val_f1', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_f1', min_delta=0, patience=1, verbose=1)
    metrics = Metrics(valid_data=[[np.array(x1_test),np.array(x2_test)], np.array(y_test)])
<<<<<<< HEAD
=======
    log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    
    tb = TensorBoard(log_dir="../logs/"+str(args.times)+"/",  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                  batch_size=32,     # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=True, # 是否可视化梯度直方图
                     write_images=True,# 是否可视化参数
                     embeddings_freq=0, 
                     embeddings_layer_names=None, 
                     embeddings_metadata=None)
>>>>>>> 优化了代码
    model.fit([np.array(x1_train),np.array(x2_train)], np.array(y_train),
             batch_size=args.batch_size,
             epochs=args.epoch,
             verbose=1,
             callbacks=[checkpoint,early_stopping,metrics],
             validation_data=[[np.array(x1_test),np.array(x2_test)], np.array(y_test)]
             )
    model.save("../models/"+str(args.times)+"/bert.h5")
def predict():
<<<<<<< HEAD
    model=build_bert(args.nclass,args.config_path,args.checkpoint_path)
=======
    model=build_bert(args)
>>>>>>> 优化了代码
    model.load_weights(args.finetuned_path+"bert.h5")

    data,label = get_data(args.test_path)
    x1_train,x2_train=get_bert_input(data,args.vocab_path,args.maxlen)

    predictions = model.predict([np.array(x1_train),np.array(x2_train)])
    detailed_predictions(predictions)#记录softmax后预测概率原始值
    final_result = predictions.argmax(axis=-1)
    write_csv(final_result)

def write_csv(result):#将预测结果写入/sub目录下csv文件
    id = pd.read_csv(args.submision_sample_path)[["id"]]
<<<<<<< HEAD
    result = pd.DataFrame(result,columns = ["label"])
    result = id.join(result)
    result.to_csv(args.submision_path,index = False)
=======
    result = pd.DataFrame(result,columns = ["y"])
    result = id.join(result)
    result.to_csv(args.sub_path+"result.csv",index = False)
>>>>>>> 优化了代码
def detailed_predictions(predictions):
    f = open("../subs/"+str(args.times)+"/predictions.txt","w",encoding = "utf8")
    for each in predictions:
        tem = [str(i) for i in each]
        f.write(" ".join(tem))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    if args.do_train == True:
       train()
    if args.do_predict == True:
       predict()
    
    
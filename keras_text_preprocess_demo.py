from keras.preprocessing import text
from keras.preprocessing import sequence
from nltk import word_tokenize
import numpy

def ids_to_tokens(id2token,text): # id2token中，固定保留0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'
    if isinstance(text, list) or isinstance(text, numpy.ndarray):
        text = [" ".join([id2token[id] for id in line if id not in [0,1,2,3]]) for line in text]
    elif isinstance(text, str):
        text = " ".join([id2token[id] for id in text if id not in [0,1,2,3]])
    return text
#keras下的Tokenizer类没有预定义 PAD、BOS、EOS、UNK等特殊字符，所以word_index,index_words成员变量需要手动更新进行添加
def preprocess_data(PathOrList,lower = True):
    if isinstance(PathOrList, list): #已经加载好了字符串
        PathOrList =[ word_tokenize(each) for each in PathOrList]
    elif isinstance(PathOrList,str): #传入需要加载文件的路径
        PathOrList = open(PathOrList,encoding = "utf8").readlines()
        PathOrList =[ word_tokenize(each.strip()) for each in PathOrList]
    maxlen = max([len(each) for each in PathOrList])

    tok = text.Tokenizer(lower = lower,filters = "")
    tok.fit_on_texts(PathOrList)

    vocab = {k:(v+3) for k,v in tok.word_index.items()}
    vocab["<pad>"] = 0
    vocab["<bos>"] = 1
    vocab["<eos>"] = 2
    vocab["<unk>"] = 3  # unknown

    renewed_id2token = {v:k for k,v in vocab.items()}

    tok.word_index = vocab

    sequences = tok.texts_to_sequences(PathOrList)
    #padded_sequences 的数据类型是np.ndaraay
    padded_sequences = sequence.pad_sequences(sequences,maxlen,value=vocab["<pad>"],dtype = "int8",padding="post")
    return vocab, renewed_id2token,padded_sequences
vocab, id2token,padded_sequences = preprocess_data(["nlp is boring"])
print(vocab)
print(id2token)
print(padded_sequences)
print(ids_to_tokens(id2token,padded_sequences))

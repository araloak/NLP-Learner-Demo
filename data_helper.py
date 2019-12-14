import numpy as np
import re
import itertools
import sys
import os 

from keras.utils import to_categorical
from collections import Counter

num_of_differernt_calsses= len(list(os.walk(file_path))[0][2])
non_chinese_sym=['\'',' ','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','z', 'x', 'c', 'v', 'b', 'n', 'm', 'l', 'k', 'j', 'h', 'g', 'f', 'd', 's', 'a', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #这是只保留汉字的版本
    string=re.sub(r'[^\u4e00-\u9fa5\\`]', ' ', string)  
    
    #这是只保留汉字、英文字母（包括'和`）、数字的版本
    #string=re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5\'\`]', ' ', string)
    
    #这是保留常用标点符号的版本
    #string=re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5(),!?":(),!?\'\`。？！]', ' ', string) #保留字母、数字、汉字和标点符号(),.!?":   
    #对英文缩略词的处理，缩写单词划分为两个单词，例："haven't"  =》 "have n't"
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    #对英文标点符号的处理,将标点符号和文本之间以空格隔开
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\.", " . ", string)
    #对中文标点符号的处理,将标点符号和文本之间以空格隔开
    string = re.sub(r"。", " 。 ", string)
    string = re.sub(r"！", " ！ ", string)
    string = re.sub(r"？", " ？ ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    #对连续出现至少两次的任意空白符的处理
    string = re.sub(r"\s{2,}", " ", string)
    string=string.replace('\\','')
    string = string.strip().lower()
    # 以下将中文字符以空格划分，便于形成字典
    new_text=""
    for i in string:
        if i not in non_chinese_sym:
            new_text+=" "+i+ " "
        else:
            new_text+=i
    return new_text
#实现训练数据的加载工作
def load_data_and_labels():
    """
    根据数据存储方式自定义读取，实现文本清洗
    """
    
    #one-hot编码
    label = to_categorical(data)

    return [text, label]

#实现将句子长度一致化——长度等于最长的句子长度，剩余的用“”\<PAD/>“补齐
def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences) # 计算最大句子长度
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

#建立词典便于之后建立字与字向量的对应关系
def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

#实现将字符组成的sentence列表转化为由vocabulary中字符索引组成的数字列表
def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

 #实现从数据的加载，到返回所有数字化处理的矩阵以及查询词典。
def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

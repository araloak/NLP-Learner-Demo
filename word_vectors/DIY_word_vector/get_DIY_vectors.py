from gensim.models import word2vec
import gensim
import logging
import sys
import os
import numpy as np
if __name__=="__main__":
    save_model_name = sys.path[0]+'\my_diy_language_model'

    # 加载已训练好的模型
    language_model = word2vec.Word2Vec.load(save_model_name)
    #--------------------------------------------------------------------------------------------------------
    # 得到当前文本text的词向量矩阵  text=[sentence,sentence,sentence...],sentence=[word,word,word,...]  
    # word为单个字符串 & 每个sentence要事先处理为等长
    for sentence in text:
        sentence_vec=[]
        for word in sentence:
            sentence_vec.append(language_model[word])
        text_vec.append(sentence_vec)
    matrix=np.array(text_vec)
    #--------------------------------------------------------------------------------------------------
    # 计算两个词的相似度/相关程度
    print(language_model.similarity("计算机", "科学"))
    print(language_model.similarity("计算机", "飞机"))
    #--------------------------------------------------------------------------------------------------
    # 计算某个词的相关词列表
    y2 = language_model.most_similar("计算机", topn=10)  # 10个最相关的
    print(u"和计算机最相关的词有：\n")
    for each in y2:
        print(each[0], each[1])

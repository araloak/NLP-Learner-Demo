from gensim.models import word2vec
import gensim
import logging
import sys
import os

if __name__=="__main__":
    save_model_name = sys.path[0]+'\my_diy_language_model'

    # 加载已训练好的模型
    model_1 = word2vec.Word2Vec.load(save_model_name)
    
    # 计算两个词的相似度/相关程度
    y1 = model_1.similarity("计算机", "科学")
    print(y1)
    y1 = model_1.similarity("计算机", "飞机")
    print(y1)

    # 计算某个词的相关词列表
    y2 = model_1.most_similar("计算机", topn=10)  # 10个最相关的
    print(u"和计算机最相关的词有：\n")
    for item in y2:
        print(item[0], item[1])
    print("-------------------------------\n")
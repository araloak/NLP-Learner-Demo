import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from gensim.models import Word2Vec
import gensim
from nlpaug.util import Action
text = '我 讨 厌 吃 西红柿'#中文文本需要分词以后不同的词之间空格连接（使用bert时不需要分词不需要空格）


'''
#使用预训练语言模型进行insert
aug = naw.ContextualWordEmbsAug(
    model_path='D:/codes/Bert_projects/pre_trained_models/chinese_L-12_H-768_A-12/pytorch_version/', action="insert")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''
'''
#使用预训练语言模型进行substitute
aug = naw.ContextualWordEmbsAug(
    model_path='D:/codes/Bert_projects/pre_trained_models/chinese_L-12_H-768_A-12/pytorch_version/', action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''

#使用word2vec进行text augment需要将中文文本分词以后每个词之间空格连接
'''
#使用word2vec语言模型计算语义相似度
model = gensim.models.KeyedVectors.load_word2vec_format(r"D:\codes\word2vector/baike_26g_news_13g_novel_229g.bin", binary=True)
print("之uhi")
print(model["智慧"])
a = model.most_similar('智慧', topn=5)
print(a)
'''

'''
#使用word2vec语言模型进行insert # model_type: word2vec, glove or fasttext
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path=r"D:\codes\word2vector/baike_26g_news_13g_novel_229g.bin",
    action="insert")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''

'''
#使用word2vec语言模型进行substitute # model_type: word2vec, glove or fasttext
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path=r"D:\codes\word2vector/baike_26g_news_13g_novel_229g.bin",
    action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''



#使用wordnet进行text augment需要将中文文本分词以后每个词之间空格连接
'''
#Substitute word by antonym
aug = naw.AntonymAug(lang = "cmn")
_text = '我喜欢黑'
augmented_text = aug.augment(_text)
print("Original:")
print(_text)
print("Augmented Text:")
print(augmented_text)
'''
'''
#Substitute word by WordNet's synonym
aug = naw.SynonymAug(aug_src='wordnet',lang="cmn")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''
'''
#Swap word randomly
aug = naw.RandomWordAug(action="swap")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''
'''
#Delete word randomly
aug = naw.RandomWordAug()
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''

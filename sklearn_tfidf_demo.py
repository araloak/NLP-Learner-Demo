#计算tfidf时可以不考虑标点符号因为基本每个doc都有
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'this is the first document ',
    'this is the second second document',
    'and the third 12 one',
    'is this the first document'
]


tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)

# 得到语料库所有词的list
print(tfidf_vec.get_feature_names())

# 得到字典
print(tfidf_vec.vocabulary_)

# 得到每个句子所对应的tfidf向量
# 向量里数字的顺序是按照词语的id顺序来的
print(tfidf_matrix.toarray())


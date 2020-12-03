import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
import os,glob
from nltk import word_tokenize
documents = ["The Saudis are preparing a report that will acknowledge that",
             "Saudi journalist Jamal Khashoggi's death was the result of an",
             "interrogation that went wrong, one that was intended to lead",
             "to his abduction from Turkey, according to two sources."]

documents_2 = ["One source says the report will likely conclude that",
                "the operation was carried out without clearance and",
                "transparency and that those involved will be held",
                "responsible. One of the sources acknowledged that the",
                "report is still being prepared and cautioned that",
                "things could change."]
#根据句子列表创建字典对象
texts = [word_tokenize(line) for line in documents ]
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)

text_2 = [word_tokenize(line) for line in documents_2]
dictionary.add_documents(text_2) #更新字典对象

#从单一文件中创建字典
dictionary = corpora.Dictionary(simple_preprocess(line,deacc=True) for line in open("python.txt"))
print(dictionary.token2id) #输出完整dict

#从dirname目录下多个文件中创建字典
class ReadTxtFiles(object):
    def __init__(self,dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in glob.glob(self.dirname+"*.txt"):
            for line in open(fname,encoding = "utf8"):
                yield simple_preprocess(line)
texts_path = r"D:\projects\tian\texar-pytorch\examples\transformer\data\cmnraw_engbpe\\"
dictionary = corpora.Dictionary(ReadTxtFiles(texts_path))
print(dictionary.token2id)

# 构建doc2bow: [(word_id,count),...]
my_doc = ["it's silly to fall in love with a stright boy.","I hate you so much that I want to kill you!"]
tokenized_list = [word_tokenize(line) for line in my_doc]
mydict = corpora.Dictionary()

mycorpus = [mydict.doc2bow(doc,allow_update=True) for doc in tokenized_list]
print(mydict.token2id)
print(mycorpus)

word_counts = [[(mydict[id],count) for id,count in line] for line in mycorpus]
print(word_counts)

#从一个文件中一行一行读入内存
class BoWCorpus(object):
    def __init__(self,path,dictionary):
        self.filepath = path
        self.dictionary = dictionary
    def __iter__(self):
        global mydict # 只有允许mydict更新的时候才需要定义为全局变量
        for line in smart_open(self.filepath,encoding = "utf8"):
            tokenized_list = simple_preprocess(line,deacc=True)
            bow = self.dictionary.doc2bow(tokenized_list,allow_update=True)
            mydict.merge_with(self.dictionary)
            yield bow
mydict = corpora.Dictionary()
bow_corpus = BoWCorpus("python.txt",dictionary=mydict)
for line in bow_corpus:
    print(line)

#保存&加载字典和doc2wo语料
mydict.save("mydict.dict")
mydict = corpora.Dictionary.load("mydict.dict")
corpora.MmCorpus.serialize("bow_corpus.mm",bow_corpus) #保存(token，count）corpus
corpus = corpora.MmCorpus("bow_corpus.mm")

import nltk

#寻找同义、反义词
from nltk.corpus import wordnet
syns = wordnet.synsets("happy")
synonyms, antonyms= [], []
for each in syns:
    for l in each.lemmas():
        print(l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(antonyms) # 反义词
print(synonyms) # 同义词

#分词、分句
from nltk.tokenize import word_tokenize, sent_tokenize
text = "I want to eat 8 sandwitchs!!! Because I'm hungry"
words = word_tokenize(text)
sents = sent_tokenize(text)

#西班牙语分词、分句
from nltk.tokenize.toktok import ToktokTokenizer
toktok = ToktokTokenizer()
sent = u"¿Quién eres tú? ¡Hola! ¿Dónde estoy?"
print(toktok.tokenize(sent))
sentences = u"¿Quién eres tú? ¡Hola! ¿Dónde estoy?"
sents = [toktok.tokenize(sent) for sent in sent_tokenize(sentences, language='spanish')]
print(sents)

#输出n-grams
print(list(nltk.bigrams(words)))
print(list(nltk.trigrams(words)))
print(list(nltk.ngrams(words,4,pad_left=True,pad_right=False))) # 4-grams

#绘制词频分布图
fd = nltk.FreqDist(words)
fd.plot()

#寻找词根
from nltk.stem import WordNetLemmatizer
words = ["run","ran","running","runs"]
wnl = WordNetLemmatizer()
for word in words:
    root_word = wnl.lemmatize(word)
    print(root_word)

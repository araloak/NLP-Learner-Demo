'''
源教程来自:

https://github.com/rockyzhengwu/FoolNLTK/blob/master/README_CH.md

'''
import fool
path=r"C:\Users\lenvov\Desktop\my_diy_dic.txt" #txt文件保存用户本地自定义词典，每行格式为：词 权重
fool.load_userdict(path) #加载自定义词典
#词典只能定义词的权值，不能定义词的词性，故对词性标注没有帮助
#fool.delete_userdict(); #删除用户自定义词典

text="习近平觉得张构架的趣多多比希斯罗机场的巧克力味的奥利奥要贵得多。"
words, ners = fool.analysis(text) #words列表保存分词后词性标注的结果（只使用自带词典不添加自定义词典），ners保存识别得到的实体（存在分词不准确但命名实体识别正确的现象，但使用自定义字典以后便可修正）
# 实体识别过程得到的words列表不受自定义词典影响。一般不用

print('文本切分：',fool.cut(text),'\n')
print('文本切分后进行词性标注：',fool.pos_cut(text),'\n')
print('words：',words,'\n')
print('实体识别',ners,'\n')

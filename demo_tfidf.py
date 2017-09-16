import jieba 
import os

textfilelist = os.listdir('text_dir')

r = re.compile(r'\(text(\d+)\)')

def key_func(m):
    return int(r.search(m).group(1))


textfilelist.sort(key=key_func)

raw_documents = []

#read file context to raw_documents list
for text in textfilelist:
    with open('text_dir/' + text) as f:
        raw_documents.append(f.read())

stopwords = {}.fromkeys(['，', '”' ,'\n','。','、','·'])


#分词处理
corpora_documents = []
for item_text in raw_documents:
    item_seg = list(jieba.cut(item_text))
    final_seg = []
    for seg in item_seg:
        if seg not in stopwords:
            final_seg.append(seg)
    corpora_documents.append(item_seg)

# 生成字典和向量语料  
dictionary = corpora.Dictionary(corpora_documents)

'''
dictionary.save('dict.txt') #保存生成的词典  
dictionary = Dictionary.load('dict.txt') #加载 
'''

#将用字符串表示的文档转换为用id表示的文档向量
corpus = [dictionary.doc2bow(text) for text in corpora_documents]

'''
corpora.MmCorpus.serialize('corpuse.mm',corpus)#保存生成的corpus  
corpus = corpora.MmCorpus('corpuse.mm')#加载  
'''

#基于这些“训练文档”计算一个TF-IDF模型
tfidf_model = models.TfidfModel(corpus)  
corpus_tfidf = tfidf_model[corpus] 

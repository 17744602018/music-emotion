
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba,os,csv
import numpy as np


file_path = "/Users/yiyizhang/Desktop/PMEmo2019/clyrics/"

TEXT_FEATURE_NUM = 100

def cut_word(sent):
    return " ".join(list(jieba.cut(sent)))


def lyric_tf_idf(i):
    X = None
    path = "{}{}.lrc".format(file_path,i)
    if os.path.exists(path):
        with open(path, "r") as rf:
            data = [line.replace("\n", "") for line in rf.readlines()]
        data_len = len(data)
        lis = []
        s1 = ""
        for temp in data[0:data_len//3]:
            s1 += cut_word(temp)

        s2 = ""
        for temp in data[data_len//3:data_len*2//3]:
            s2 += cut_word(temp)

        s3 = ""
        for temp in data[data_len*2//3:]:
            s3 += cut_word(temp)

        lis.append(s1)
        lis.append(s2)
        lis.append(s3)
        transfer = TfidfVectorizer(max_features=TEXT_FEATURE_NUM)
        X = transfer.fit_transform(lis).toarray()
    return X 

import numpy
from cnn.cnn import NeuralNetwork
from anotation_pic.pic1 import DrawPic
from utils import Utils
from sklearn.metrics import  recall_score, f1_score, precision_score
from sklearn.metrics import classification_report
from lstm.audiolyriclstm import AudioLyricLSTM

def bpnn():


    # data_file_path = "/Users/yiyizhang/Desktop/PMEmo2019/extractfeature/voice_feature.csv"
    data_file_path = "fusiondata.csv"
        
    input_nodes=500
    hidden_nodes=200
    output_nodes=4
            
    #学习率
    learning_rate=0.1
            
    n=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
            
    #打开文件并获取其中的内容
    training_data_file=open(data_file_path,'r')
    training_data_list=training_data_file.readlines()
    #关闭文件
    training_data_file.close()
    print(len(training_data_list))
    #5个世代，训练一次称为一个世代
    epochs=5
            
    #训练五次
    for e in range(epochs):
        #遍历所有训练集中的数据
        for record in training_data_list:
            #接收record，根据逗号，将这一长串进行拆分，split()函数是执行拆分任务的，其中有一个参数告诉函数根据哪个符号进行拆分
            all_values=record.split(',')
            #乘以0.99，变为0.0到0.99，之后再加上0.01，得到0.01到1.00
            inputs=(numpy.asfarray(all_values[:-1])*0.99)+0.01
            #使用numpy.zeros()创建0填充的数组
            targets=numpy.zeros(output_nodes)+0.01
            targets[int(all_values[-1])-1]=0.99
            # print(inputs) one-hot  2  0100 1000 0010 0001
            # print(targets)#[0.01 0.01 0.01 0.01 0.01 0.99 0.01 0.01 0.01 0.01]
            #训练数据
            n.train(inputs,targets)


    #获取测试记录
    #测试神经网络
    test_data_file=open(data_file_path,'r')
    test_data_list=test_data_file.readlines()
    test_data_file.close()
        
    #记分卡，在每条测试之后都会更新
    scorecard=[]

    y_p = []
    y_t = []
    for record in test_data_list:
        #拆分文本
        all_values=record.split(',')
        #记下标签
        correct_label=int(all_values[-1])-1
        #调整剩下的值，使之适合查询神经网络
        inputs=(numpy.asfarray(all_values[:-1]) * 0.99)+0.01
        #将神经网络的回答保存在outputs中
        outputs=n.query(inputs)
        #numpy.argmax()函数可以找出数组的最大值，并告诉我们最大值的位置
        label=numpy.argmax(outputs)
        # print("correct:{},predict:{}".format(correct_label,label))
        #将计算得出的标签与已知正确标签对比，如果相同，在记分表后面加1，如果不同，在记分表后面，加0
        if(label==correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
        y_p.append(label)
        y_t.append(correct_label)
    print(y_t)
    print(classification_report(y_t, y_p))
    print("auc",Utils().multiclass_roc_auc_score(y_t,y_p))

    
    
    

def draw():
    fields, data = Utils.get_csv_dic_list("/Users/yiyizhang/Desktop/PMEmo2019/new_anotations/static_annotations.csv")

    mean_x = []
    mean_y = []
    for item in data:
        mean_x.append(float(item['mean_arousal']))
        mean_y.append(float(item['mean_valence']))
    # print(len(mean_x))
    # print(len(mean_y))
    DrawPic.draw(mean_x,mean_y)


# bpnn()
# from ml.svm import *
# from ml.forest import * 
# from ml.gaussianNB import * 
# from ml.logistic import * 
# from dataload.loaddata import * 
# from dataload.fusiondata import * 
# from dataload.randompickd import * 
# from MFCC.mfcc import *
# from lstm.test import * 
# print(len(datas))
# print(len(labels))
# print(len(datas[0]))

audio_lyric_lstm = AudioLyricLSTM(150,300,True)
def xx():
    import numpy as np
    import pandas as pd
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, concatenate
    from keras.callbacks import EarlyStopping

    # 加载数据，这里以随机生成的数据作为例子
    X_lyric_train = np.random.rand(100, 10, 20)  # 假设歌词数据的维度是(100, 10, 20)
    X_audio_train = np.random.rand(100, 50, 30)  # 假设音频数据的维度是(100, 50, 30)
    y_train = np.random.randint(0, 4, size=100)  # 假设标签有4类，标签的维度是(100,)
    print(X_lyric_train.shape)
    print(X_audio_train.shape)
    print(y_train.shape)
    # 定义模型的输入层
    lyric_input = Input(shape=(X_lyric_train.shape[1], X_lyric_train.shape[2]))
    audio_input = Input(shape=(X_audio_train.shape[1], X_audio_train.shape[2]))

    # 定义LSTM层
    lstm_lyric = LSTM(64)(lyric_input)
    lstm_audio = LSTM(64)(audio_input)

    # 合并LSTM层的输出
    merged = concatenate([lstm_lyric, lstm_audio])

    # 定义输出层
    output = Dense(4, activation='softmax')(merged)

    # 定义模型
    merged_model = Model(inputs=[lyric_input, audio_input], outputs=output)

    # 编译模型
    merged_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 定义EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # 训练模型
    history = merged_model.fit(x=[X_lyric_train, X_audio_train], y=y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

    # 在测试集上测试模型
    X_lyric_test = np.random.rand(20, 10, 20)
    X_audio_test = np.random.rand(20, 50, 30)
    y_test = np.random.randint(0, 4, size=20)
    test_loss, test_acc = merged_model.evaluate(x=[X_lyric_test, X_audio_test], y=y_test)

    print('Test accuracy:', test_acc)


def xxx():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from wordcloud import WordCloud

    # 加载数据
    data = pd.read_csv("/Users/yiyizhang/Desktop/music-emotion/data/text_feature_lyric.csv")

    # 使用 TF-IDF 提取特征
    tfidf = TfidfVectorizer(stop_words='english')
    features = tfidf.fit_transform(data["text"])

    # 获取每个单词的权重
    weights = np.asarray(features.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tfidf.get_feature_names_out(), 'weight': weights})

    # 生成词云
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(weights_df.set_index('term')['weight'].to_dict())

    # 展示词云
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
# xxx()
import numpy
from cnn.cnn import NeuralNetwork
from anotation_pic.pic1 import DrawPic
from utils import Utils
from sklearn.metrics import  recall_score, f1_score, precision_score

def bpnn():

    data = []

    data_file_path = "/Users/yiyizhang/Desktop/PMEmo2019/extractfeature/text_feature.csv"
        
    input_nodes=300
    hidden_nodes=100
    output_nodes=4
            
    #学习率
    learning_rate=0.1
            
    n=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
            
    #打开文件并获取其中的内容
    training_data_file=open(data_file_path,'r')
    training_data_list=training_data_file.readlines()
    #关闭文件
    training_data_file.close()
            
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
    print('Recall:', recall_score(y_t, y_p, average='macro'))
    print('F1-score:', f1_score(y_t, y_p, average='macro'))
    print('Precision score:', precision_score(y_t, y_p, average='macro'))
    #计算准确率
    scorecard_array=numpy.asarray(scorecard)
    #转换为数组
    print("performance = ",scorecard_array.sum() / scorecard_array.size)

    data.append(recall_score(y_t, y_p, average='macro'))
    data.append(f1_score(y_t, y_p, average='macro'))
    data.append(precision_score(y_t, y_p, average='macro'))
    data.append(scorecard_array.sum() / scorecard_array.size)
    DrawPic.draw_histogram("BPNN",data,['Recall','F1-score','Precision score','performance'])


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

bpnn()

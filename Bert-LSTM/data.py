import numpy as np

"""
获取训练数据和测试数据
"""
class TrainTestData():


    @staticmethod
    def get_train_data_lable(filename):
        data = []
        labels = []
        datafile = open(filename)
        for line in datafile:
            row = []
            fields = line.strip().split(',')
            for field in fields[:-1]:
                row.append(float(field))
            data.append(row)
            labels.append(fields[-1])
        data = np.array(data)
        labels = np.array(labels)
        return data, labels

 
    @staticmethod
    def get_test_data_lable(filename):
        data = []
        labels = []
        datafile = open(filename)
        for line in datafile:
            row = []
            fields = line.strip().split(',')
            for field in fields[:-1]:
                row.append(float(field))
            data.append(row)
            labels.append(fields[-1])
        data = np.array(data)
        labels = np.array(labels)
        return data, labels 
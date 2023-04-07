# 导入所需的库和模型：
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from data import TrainTestData
#加载预训练的 Bert 模型和分词器：
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
#定义模型结构，包括 Bert 模型和 LSTM 模型：
model = Sequential()
model.add(bert_model)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
print("sfdsf")
#编译模型并指定优化器、损失函数和评价指标：
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#准备训练数据，并将其转换为 Bert 输入格式：
inputs = tf.keras.Input(shape=(128,), dtype=tf.int32, name='input_word_ids')
outputs = bert_model(inputs)[1]
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)
# 准备训练数据 # 准备训练标签
train_data ,train_labels = TrainTestData().get_train_data_lable("/Users/yiyizhang/Desktop/PMEmo2019/extractfeature/text_feature_lyric.csv")

train_tokens = []
train_masks = []
for text in train_data:
    encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_attention_mask=True, return_tensors='tf')
    train_tokens.append(encoded_dict['input_ids'])
    train_masks.append(encoded_dict['attention_mask'])

train_tokens = tf.concat(train_tokens, axis=0)
train_masks = tf.concat(train_masks, axis=0)

model.fit([train_tokens, train_masks], train_labels, epochs=10, batch_size=32)
#对模型进行评估和预测：
# 准备测试数据
# 准备测试标签
test_data ,test_labels = TrainTestData().get_train_data_lable("/Users/yiyizhang/Desktop/PMEmo2019/extractfeature/text_feature_lyric.csv")

test_tokens = []
test_masks = []
for text in test_data:
    encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_attention_mask=True, return_tensors='tf')
    test_tokens.append(encoded_dict['input_ids'])
    test_masks.append(encoded_dict['attention_mask'])

test_tokens = tf.concat(test_tokens, axis=0)
test_masks = tf.concat(test_masks, axis=0)

loss, accuracy = model.evaluate([test_tokens, test_masks], test_labels, batch_size=32)
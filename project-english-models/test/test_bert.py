import torch
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train_file_path = '/userhome/cs2/u3603202/nlp/project/twitter_training.csv'
val_file_path = '/userhome/cs2/u3603202/nlp/project/twitter_validation.csv'
# test_file_path = '/content/hku.csv'

train_data = pd.read_csv(train_file_path, header=None)
val_data = pd.read_csv(val_file_path, header=None)
# test_data = pd.read_csv(test_file_path, header=None)

"""
加载数据
5个测试集对应5个label，即5种sentiment
"""
# 载入数据，建立标签，部分当作
train_data.reset_index(drop=True,inplace=True)
val_data.reset_index(drop=True,inplace=True)
df = pd.concat([train_data,val_data], axis=0)
df.drop([0], axis=1, inplace=True)
df.columns = ['platform','sentiment','text']
df.drop(['platform'], axis=1, inplace=True)
df.sentiment = df.sentiment.map({"Neutral":0, "Irrelevant":0 ,"Positive":1,"Negative":2})
df.dropna(inplace=True)


# 连接每个数据集作为训练集
train_data, val_data = train_test_split(df,test_size = 0.2, stratify = df.sentiment.values)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# 加载bert的tokenize方法
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True
        )

        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


"""
在tokenize之前，我们需要指定句子的最大长度。
"""

# Encode 我们连接的数据
encoded_comment = [tokenizer.encode(sent, add_special_tokens=True) for sent in df.text.values]

# 找到最大长度
max_len = max([len(sent) for sent in encoded_comment])
print('Max length: ', max_len)

MAX_LEN = 315
train_inputs, train_masks = preprocessing_for_bert(train_data.text.values)
test_inputs, test_masks = preprocessing_for_bert(val_data.text.values)
print(train_inputs.shape)

train_labels = torch.tensor(train_data.sentiment.values)
test_labels = torch.tensor(val_data.sentiment.values)




"""Create PyTorch DataLoader

我们将使用torch DataLoader类为数据集创建一个迭代器。这将有助于在训练期间节省内存并提高训练速度。

"""
# 转化为tensor类型




# 用于BERT微调, batch size 16 or 32较好.
batch_size = 16

# 给训练集创建 DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# print(train_dataloader)

# 给验证集创建 DataLoader
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)



# 自己定义的Bert分类器的类，微调Bert模型
class BertClassifier(nn.Module):
    def __init__(self, ):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 100, 3

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits


# 注意这个地方的logits是全连接的返回， 两个output就是01二分类，我们这里用的是ouput为3，就是老师所需要的三分类问题


"""
然后就是深度学习的老一套定义优化器还有学习率等
"""


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list, save_path: str):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.save_path = save_path

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        plt.savefig(save_path)

def initialize_model(model_path, epochs=2):
    """
    初始化我们的bert，优化器还有学习率，epochs就是训练次数
    """
    # 初始化我们的Bert分类器
    bert_classifier = BertClassifier()
    bert_classifier.load_state_dict(torch.load(model_path))
    # 用GPU运算
    bert_classifier.to(device)
    # 创建优化器
    return bert_classifier


# 训练模型
        # =======================================
        #               Evaluation
        # =======================================

# 在测试集上面来看看我们的训练效果
def evaluate(model, test_dataloader, confusion):
    """
    在每个epoch后验证集上评估model性能
    """
    # model放入评估模式
    model.eval()

    # 准确率和误差
    test_accuracy = []
    test_loss = []

    # 验证集上的每个batch
    for batch in test_dataloader:
        # 放到GPU上
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # 计算结果，不计算梯度
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)  # 放到model里面去跑，返回验证集的ouput就是一行三列的
            # label向量可能性，这个时候还没有归一化所以还不能说是可能性，反正归一化之后最大的就是了

        # get预测结果，这里就是求每行最大的索引咯，然后用flatten打平成一维
        preds = torch.argmax(logits, dim=1).flatten()  # 返回一行中最大值的序号

        # 计算准确率，这个就是俩比较，返回相同的个数, .cpu().numpy()就是把tensor从显卡上取出来然后转化为numpy类型的举证好用方法
        # 最后mean因为直接bool形了，也就是如果预测和label一样那就返回1，正好是正确的个数，求平均就是准确率了
        confusion.update(preds.to("cpu").numpy(), b_labels.to("cpu").numpy())

    # 计算整体的平均正确率和loss
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy


# 先来个两轮
model_path = "/userhome/cs2/u3603202/nlp/project/output/0.pth"
save_path = "/userhome/cs2/u3603202/nlp/project/output/bert/matrix.png"
bert_classifier = initialize_model(model_path, epochs=2)
# print("Start training and validation:\n")
print("Start training and testing:\n") # 这个是有评估的
label_dict = {0: "Neutral", 1: "Positive", 2: "Negative"}
confusion = ConfusionMatrix(num_classes=3, labels=label_dict, save_path = save_path)
evaluate(bert_classifier, test_dataloader, confusion)
confusion.plot()
confusion.summary()




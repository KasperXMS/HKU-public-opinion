import torch
import time
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import XLNetModel, XLNetTokenizer
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
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)


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

MAX_LEN = 398
train_inputs, train_masks = preprocessing_for_bert(train_data.text.values)
test_inputs, test_masks = preprocessing_for_bert(val_data.text.values)

train_labels = torch.tensor(train_data.sentiment.values)
test_labels = torch.tensor(val_data.sentiment.values)




"""Create PyTorch DataLoader

我们将使用torch DataLoader类为数据集创建一个迭代器。这将有助于在训练期间节省内存并提高训练速度。

"""
# 转化为tensor类型




# 用于BERT微调, batch size 16 or 32较好.
batch_size = 8

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

        self.bert = XLNetModel.from_pretrained('xlnet-base-cased')

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



def initialize_model(epochs=2):
    """
    初始化我们的bert，优化器还有学习率，epochs就是训练次数
    """
    # 初始化我们的Bert分类器
    bert_classifier = BertClassifier()
    # 用GPU运算
    bert_classifier.to(device)
    # 创建优化器
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # 默认学习率
                      eps=1e-8  # 默认精度
                      )
    # 训练的总步数
    total_steps = len(train_dataloader) * epochs
    # 学习率预热
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


# 实体化loss function
loss_fn = nn.CrossEntropyLoss()  # 交叉熵


# 训练模型
def train(model, save_path, train_dataloader, test_dataloader=None, epochs=2, evaluation=False):
    # 开始训练循环
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # 表头
        print(f"{'Epoch':^7} | {'每40个Batch':^9} | {'训练集 Loss':^12} | {'测试集 Loss':^10} | {'测试集准确率':^9} | {'时间':^9}")
        print("-" * 80)

        # 测量每个epoch经过的时间
        t0_epoch, t0_batch = time.time(), time.time()

        # 在每个epoch开始时重置跟踪变量
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # 把model放到训练模式
        model.train()

        # 分batch训练
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # 把batch加载到GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            #print(b_labels.shape)
            # 归零导数
            model.zero_grad()
            # 真正的训练
            logits = model(b_input_ids, b_attn_mask)
            #print(logits.shape)
            # 计算loss并且累加

            loss = loss_fn(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()
            # 反向传播
            loss.backward()
            # 归一化，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数和学习率
            optimizer.step()
            scheduler.step()

            # Print每40个batch的loss和time
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # 计算40个batch的时间
                time_elapsed = time.time() - t0_batch

                # Print训练结果
                print(
                    f"{epoch_i + 1:^7} | {step:^10} | {batch_loss / batch_counts:^14.6f} | {'-':^12} | {'-':^13} | {time_elapsed:^9.2f}")

                # 重置batch参数
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 计算平均loss 这个是训练集的loss
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 80)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:  # 这个evalution是我们自己给的，用来判断是否需要我们汇总评估
            # 每个epoch之后评估一下性能
            # 在我们的验证集/测试集上.
            test_loss, test_accuracy = evaluate(model, test_dataloader)
            # Print 整个训练集的耗时
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^10} | {avg_train_loss:^14.6f} | {test_loss:^12.6f} | {test_accuracy:^12.2f}% | {time_elapsed:^9.2f}")
            print("-" * 80)
            torch.save(model.state_dict(), os.path.join(save_path, str(epoch_i) + ".pth"))
        print("\n")


# 在测试集上面来看看我们的训练效果
def evaluate(model, test_dataloader):
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

        # 计算误差
        loss = loss_fn(logits, b_labels.long())
        test_loss.append(loss.item())

        # get预测结果，这里就是求每行最大的索引咯，然后用flatten打平成一维
        preds = torch.argmax(logits, dim=1).flatten()  # 返回一行中最大值的序号

        # 计算准确率，这个就是俩比较，返回相同的个数, .cpu().numpy()就是把tensor从显卡上取出来然后转化为numpy类型的举证好用方法
        # 最后mean因为直接bool形了，也就是如果预测和label一样那就返回1，正好是正确的个数，求平均就是准确率了
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # 计算整体的平均正确率和loss
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy


# 先来个两轮
save_path = "/userhome/cs2/u3603202/nlp/project/output/XLNet"
if not os.path.exists(save_path):
    os.mkdir(save_path)
bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
# print("Start training and validation:\n")
print("Start training and testing:\n")
train(bert_classifier, save_path, train_dataloader, test_dataloader, epochs=2, evaluation=True)  # 这个是有评估的


net = BertClassifier()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))




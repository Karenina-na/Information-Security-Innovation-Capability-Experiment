import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from Models.Transformer.Coder.Modules import MultiHeadAttention, PoswiseFeedForwardNet, PositionalEncoding
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from util.train.print_loss import print_start_train, print_batch_loss, \
    print_epoch_loss, print_end_train, print_test_loss, print_grid_search_result
from DataProcess.Dataprocess import Dataset
from sklearn.model_selection import train_test_split
from util.util import normalized, standardize, oneHot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 一维数据分类模型
# position-wise feed forward net
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)  # [batch_size, seq_len, d_model]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k

    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn = softmax(Q * K^T / sqrt(d_k))
        # context = softmax(attn * V)
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # attention : [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)  # context : [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k). \
            transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k). \
            transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v). \
            transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context, attn = ScaledDotProductAttention(self.d_k).forward(Q, K, V)
        context = context.transpose(1, 2). \
            reshape(batch_size, -1, self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.layer_norm(output + residual), attn


# one decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = \
            self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, seq_size, n_layers, d_model, d_ff, d_k, d_v, n_heads, n_class):
        super(Decoder, self).__init__()
        self.embedding = nn.Sequential(
            # 卷积扩充维度 [batch, seq_len] -> [batch, seq_len, d_model]
            nn.Conv1d(seq_size, seq_size * d_model, 1)
        )
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])
        self.classciation = nn.Sequential(
            nn.Linear(d_model, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, tgt_len]
        """
        dec_inputs = self.embedding(dec_inputs.unsqueeze(2)).reshape(dec_inputs.size(0), -1, d_model)
        # dec_inputs: [batch_size, tgt_len, d_model]
        dec_self_attns, dec_enc_attns = [], []
        dec_outputs = dec_inputs
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs)
            dec_self_attns.append(dec_self_attn)

        # dec_outputs: [batch_size, tgt_len, d_model] => [batch_size, d_model]
        dec_outputs = nn.AdaptiveAvgPool1d(1)(dec_outputs.permute(0, 2, 1)).squeeze(2)
        return self.classciation(dec_outputs)


def train_transformer_decoder(bs, lr, ep, tts):
    batch_size = bs
    learning_rate = lr
    epochs = ep

    print("-" * 50)
    path = "../Data/processed_data"
    Data = Dataset(path, class_num, random_state=rand)

    # 设定随机数
    torch.manual_seed(rand)
    torch.cuda.manual_seed(rand)
    np.random.seed(rand)
    Train, Val, TrainLabel, ValLabel = \
        train_test_split(Data.data, Data.label, test_size=tts, random_state=rand, shuffle=True)
    Test = Data.testData
    TestLabel = Data.testLabel
    Train = Train.reshape(-1, Train.shape[1], 1)
    Val = Val.reshape(-1, Val.shape[1], 1)
    Test = Test.reshape(-1, Test.shape[1], 1)

    print("-" * 50)
    print("X_train.shape: ", Train.shape)
    print("X_test.shape: ", Val.shape)
    print("labels_train.shape: ", TrainLabel.shape)
    print("labels_test.shape: ", ValLabel.shape)
    print("-" * 50)

    # 数据处理
    # Train = standardize(Train)
    # Val = standardize(Val)
    # Test = standardize(Test)
    # Train = normalized(Train)
    # Val = normalized(Val)
    # Test = normalized(Test)
    Train = Train.reshape(-1, Train.shape[1] * Train.shape[2])
    Val = Val.reshape(-1, Val.shape[1] * Val.shape[2])
    Test = Test.reshape(-1, Test.shape[1] * Test.shape[2])

    # 转换为Tensor
    Train = torch.from_numpy(Train).float()
    Val = torch.from_numpy(Val).float()
    Test = torch.from_numpy(Test).float()
    TrainLabel = torch.from_numpy(TrainLabel).float()
    ValLabel = torch.from_numpy(ValLabel).float()
    TestLabel = torch.from_numpy(TestLabel).float()

    # 转换为Dataset
    TrainDataset = torch.utils.data.TensorDataset(Train, TrainLabel)
    ValDataset = torch.utils.data.TensorDataset(Val, ValLabel)
    TestDataset = torch.utils.data.TensorDataset(Test, TestLabel)

    # 转换为DataLoader
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=True)
    TestDataLoader = DataLoader(TestDataset, batch_size=batch_size, shuffle=True)

    # 模型

    model = Decoder(Train.shape[1], n_layers, d_model, d_ff, d_k, d_v, n_heads, class_num).to('cuda:0')

    # 交叉熵
    criterion = torch.nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练
    print_start_train()
    gd_acc = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(TrainDataLoader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print_batch_loss(batch_idx, loss.item())

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()

        acc = 100. * correct / total

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(ValDataLoader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, torch.max(target, 1)[1])
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == torch.max(target, 1)[1]).sum().item()
            val_loss /= len(ValDataLoader)
            val_acc = 100. * correct / total

        gd_acc.append(val_acc)

        print_epoch_loss((epoch, epoch_loss, acc, val_loss, val_acc), "accuracy")
    print_end_train()

    # 测试
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(TestDataLoader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()
        test_acc = 100. * correct / total

    print_test_loss((test_loss, test_acc), "accuracy")

    return np.mean(gd_acc), model, Pca
    # return test_acc


def train_SVM(tts):
    print("-" * 50)
    path = "../Data/processed_data"
    Data = Dataset(path, class_num, random_state=rand)

    # 设定随机数
    torch.manual_seed(rand)
    torch.cuda.manual_seed(rand)
    np.random.seed(rand)
    Train, Val, TrainLabel, ValLabel = \
        train_test_split(Data.data, Data.label_not_onehot, test_size=tts, random_state=rand, shuffle=True)
    Test = Data.testData
    TestLabel = Data.testLabel_not_onehot
    Train = Train.reshape(-1, Train.shape[1], 1)
    Val = Val.reshape(-1, Val.shape[1], 1)
    Test = Test.reshape(-1, Test.shape[1], 1)

    print("-" * 50)
    print("X_train.shape: ", Train.shape)
    print("X_test.shape: ", Val.shape)
    print("labels_train.shape: ", TrainLabel.shape)
    print("labels_test.shape: ", ValLabel.shape)
    print("-" * 50)

    # 数据处理
    # Train = standardize(Train)
    # Val = standardize(Val)
    # Test = standardize(Test)
    # Train = normalized(Train)
    # Val = normalized(Val)
    # Test = normalized(Test)
    Train = Train.reshape(-1, Train.shape[1] * Train.shape[2])
    Val = Val.reshape(-1, Val.shape[1] * Val.shape[2])
    Test = Test.reshape(-1, Test.shape[1] * Test.shape[2])

    # grid search
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    parameters = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
    }

    # SVM
    from sklearn.svm import SVC

    svm = SVC()

    cv = StratifiedKFold(n_splits=2, shuffle=True)
    grid_search = GridSearchCV(svm, parameters, cv=cv, scoring='accuracy', verbose=2)
    grid_search.fit(Train, TrainLabel)

    best_model = grid_search.best_estimator_
    print("-" * 50)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print("-" * 50)
    print("VAL Classification Report:")
    predictionsTest = best_model.predict(Val)
    print("TEST Classification Report:")
    print(classification_report(ValLabel, predictionsTest))
    print("Accuracy:", accuracy_score(ValLabel, predictionsTest))

def train_RF(tts):
    print("-" * 50)
    path = "../Data/processed_data"
    Data = Dataset(path, class_num, random_state=rand)

    # 设定随机数
    torch.manual_seed(rand)
    torch.cuda.manual_seed(rand)
    np.random.seed(rand)
    Train, Val, TrainLabel, ValLabel = \
        train_test_split(Data.data, Data.label_not_onehot, test_size=tts, random_state=rand, shuffle=True)
    Test = Data.testData
    TestLabel = Data.testLabel_not_onehot
    Train = Train.reshape(-1, Train.shape[1], 1)
    Val = Val.reshape(-1, Val.shape[1], 1)
    Test = Test.reshape(-1, Test.shape[1], 1)

    print("-" * 50)
    print("X_train.shape: ", Train.shape)
    print("X_test.shape: ", Val.shape)
    print("labels_train.shape: ", TrainLabel.shape)
    print("labels_test.shape: ", ValLabel.shape)
    print("-" * 50)

    # 数据处理
    # Train = standardize(Train)
    # Val = standardize(Val)
    # Test = standardize(Test)
    # Train = normalized(Train)
    # Val = normalized(Val)
    # Test = normalized(Test)
    Train = Train.reshape(-1, Train.shape[1] * Train.shape[2])
    Val = Val.reshape(-1, Val.shape[1] * Val.shape[2])
    Test = Test.reshape(-1, Test.shape[1] * Test.shape[2])

    # grid search
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    parameters = {
        'n_estimators': [100, 200],
    }

    # random forest
    from sklearn.ensemble import RandomForestClassifier
    RF = RandomForestClassifier()

    cv = StratifiedKFold(n_splits=2, shuffle=True)
    grid_search = GridSearchCV(RF, parameters, cv=cv, scoring='accuracy', verbose=2)
    grid_search.fit(Train, TrainLabel)

    best_model = grid_search.best_estimator_
    print("-" * 50)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print("-" * 50)
    print("VAL Classification Report:")
    predictionsTest = best_model.predict(Val)
    print("TEST Classification Report:")
    print(classification_report(ValLabel, predictionsTest))
    print("Accuracy:", accuracy_score(ValLabel, predictionsTest))

# Transformer Parameters
d_model = 128  # Embedding Size
d_ff = 1024  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
lr = 0.005  # learning rate
BatchSize = 64  # Batch size
Epoch = 100  # Epoch
class_num = 15
rand = 3407

"""
TEST Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.94      0.96       194
           1       0.97      1.00      0.98       202
           2       1.00      1.00      1.00       179
           3       1.00      1.00      1.00       205
           4       1.00      1.00      1.00       201
           5       0.99      0.99      0.99       192
           6       1.00      0.99      1.00       197
           7       0.99      1.00      1.00       213
           8       1.00      1.00      1.00       192
           9       1.00      1.00      1.00       201
          10       1.00      1.00      1.00       213
          11       0.99      0.99      0.99       195
          12       0.85      0.85      0.85       185
          13       1.00      1.00      1.00       222
          14       0.87      0.87      0.87       209

    accuracy                           0.98      3000
   macro avg       0.98      0.98      0.98      3000
weighted avg       0.98      0.98      0.98      3000

Accuracy: 0.9763333333333334
"""

if __name__ == '__main__':
    # train_transformer_decoder(BatchSize, lr, Epoch, 0.2)
    # train_SVM(0.2)
    train_RF(0.2)
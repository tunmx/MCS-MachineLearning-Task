import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

# 填补缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# 从名字中提取头衔
train_data['Title'] = train_data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
test_data['Title'] = test_data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

# 对分类变量进行独热编码
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked', 'Title'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked', 'Title'])

# 删除不相关特征
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 确保测试数据有与训练数据相同的列
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[train_data.columns.drop('Survived')]

# 提取特征和标签
X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_data['Survived']
X_test = test_data.drop(['PassengerId'], axis=1)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.layer1 = nn.Linear(X.shape[1], 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.softmax(self.layer4(x))
        return x


# 转换为张量
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_val = torch.tensor(y_val.values, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = TitanicModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_preds = torch.max(val_outputs, 1)
            val_accuracy = (val_preds == y_val).float().mean()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')


# 生成预测结果
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_preds = torch.max(test_outputs, 1)

# 创建结果文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_preds.numpy()
})
submission.to_csv('submission.csv', index=False)

print("Prediction results have been saved to 'submission.csv'.")

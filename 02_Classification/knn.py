import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
import os

root = os.getcwd()  # 获取根目录

batch_size = 100

'''下载 MNIST数据集'''
train_datasets = datasets.MNIST(root='./cache/', train=True, transform=None, download=True)
test_datasets = datasets.MNIST(root='./cache/', train=False, transform=None, download=True)

'''加载数据'''
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

'''可视化函数（可选）'''


def visualize(digit, V):
    '''
    如果不完全下载数据集, 可自行更改range范围
    '''
    assert V == 'train' or V == 'test', 'V must train or test,train代表保存可视化的训练集,test代表保存可视化的测试集'
    digit = digit.reshape(digit.shape[0], 28, 28)

    if (V == 'train'):
        pic = os.path.join(root, "visualization/train")
        if not os.path.isdir(pic):
            os.makedirs(pic)

        for i in range(0, len(digit)):
            plt.imshow(digit[i], cmap=plt.cm.binary)
            plt.savefig(pic + '/{}.png'.format(i))
    elif (V == 'test'):
        pic1 = os.path.join(root, "visualization/test")
        if not os.path.isdir(pic1):
            os.makedirs(pic1)

        for i in range(0, len(digit)):
            plt.imshow(digit[i], cmap=plt.cm.binary)
            plt.savefig(pic1 + '/{}.png'.format(i))


'''图像预处理：归一化'''


def getXmean(x_train):
    x_train = np.reshape(x_train, (x_train.shape[0], -1))  # 将28*28像素展开成一个一维的行向量
    mean_image = np.mean(x_train, axis=0)  # 求所有图片每一个像素上的平均值
    return mean_image


def centralized(x_test, mean_image):
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_test = x_test.astype(float)
    x_test -= mean_image
    return x_test


'''归一化函数'''


def normalization(pre, x_train, y_train, x_test, y_test):
    assert pre == 'Y' or pre == 'N', 'pre must Y or N,Y代表进行归一化,N代表不进行归一化'

    if (pre == 'Y'):
        mean_image = getXmean(x_train)
        x_train = centralized(x_train, mean_image)

        mean_image = getXmean(x_test)
        x_test = centralized(x_test, mean_image)

        print("train_data:", x_train.shape)  # （样本数，图片大小）
        print("train_labels:", len(y_train))  # 样本数
        print("test_data:", x_test.shape)
        print("test_labels:", len(y_test))
        return x_train, y_train, x_test, y_test

    elif (pre == 'N'):
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)  # 需要reshape之后才能放入knn分类器
        x_test = x_test.reshape(x_test.shape[0], 28 * 28)

        print("train_data:", x_train.shape)
        print("train_labels:", len(y_train))
        print("test_data:", x_test.shape)
        print("test_labels:", len(y_test))
        return x_train, y_train, x_test, y_test


'''KNN分类器'''


class Knn:
    '''
    X_train: 训练集数据
    y_train: 训练集标签
    X_test:  测试集数据
    y_test:  测试集标签
    '''

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.Xtr = X_train
        self.ytr = y_train

    def predict(self, k, dis, X_test):
        assert dis == 'E' or dis == 'M', 'dis must E or M,E代表欧拉距离,M代表曼哈顿距离'
        num_test = X_test.shape[0]
        label_list = []
        # 使用欧拉公式作为距离测量
        if dis == 'E':
            for i in tqdm.tqdm(range(num_test)):
                distances = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i],
                                                                (self.Xtr.shape[0], 1)))) ** 2, axis=1))
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                class_count = {}
                for i in topK:
                    class_count[self.ytr[i]] = class_count.get(self.ytr[i], 0) + 1
                sorted_class_count = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
                label_list.append(sorted_class_count[0][0])
            return np.array(label_list)
        # 使用曼哈顿公式进行度量
        if dis == 'M':
            for i in range(num_test):
                distances = np.sum(abs(((self.Xtr - np.tile(X_test[i],
                                                            (self.Xtr.shape[0], 1)))), axis=1))
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                class_count = {}
                for i in topK:
                    class_count[self.ytr[i]] = class_count.get(self.ytr[i], 0) + 1
                sorted_class_count = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
                label_list.append(sorted_class_count[0][0])
            return np.array(label_list)


'''KNN分类主程序'''
num_test = 200  # 测试集数量
x_train, y_train, x_test, y_test = normalization('Y', train_loader.dataset.data.numpy(),
                                                 train_loader.dataset.targets.numpy(),
                                                 test_loader.dataset.data[:num_test].numpy(),
                                                 test_loader.dataset.targets[:num_test].numpy())

KNN_classifier = Knn()
KNN_classifier.fit(x_train, y_train)

# 不同 k下的识别
for k in range(1, 6, 2):
    y_pred = KNN_classifier.predict(k, 'E', x_test)
    print(y_pred)
    # 计算识别准确率
    num_correct = np.sum(y_pred == y_test)  # 判断标签是否一致
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct when k= %d => accuracy: %f' % (num_correct, num_test, k, accuracy))

print('Training and Testing are over and the dataset is being visualized!')

# 可视化
# visualize(x_train,'train')
# visualize(x_test,'test')



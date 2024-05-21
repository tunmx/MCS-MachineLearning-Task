# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

# 加载Wine数据集
wine = datasets.load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# 将数据转换为DataFrame便于处理和可视化
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 数据概览
print("数据集概览：")
display(df.head())

# 数据统计信息
print("\n数据统计信息：")
display(df.describe())

# 类别分布
print("\n类别分布：")
sns.countplot(x='target', data=df)
plt.title('Wine Classes Distribution')
plt.show()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 初始化和训练SVM分类器（线性核）
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# 预测
y_pred = svm_classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"\n线性核SVM模型的准确率: {accuracy}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# 3D可视化分类结果
def plot_3d(X, y, model, title):
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制训练集的散点图
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis', edgecolor='k', s=40)

    # 绘制分类超平面
    xx, yy = np.meshgrid(np.linspace(X_reduced[:, 0].min(), X_reduced[:, 0].max(), num=50),
                         np.linspace(X_reduced[:, 1].min(), X_reduced[:, 1].max(), num=50))
    zz = (-model.intercept_[0] - model.coef_[0][0] * xx - model.coef_[0][1] * yy) / model.coef_[0][2]
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='blue')

    ax.set_title(title)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()


plot_3d(X_train, y_train, svm_classifier, "3D Visualization of SVM Classification on Wine Data (Training Set)")

# 尝试其他核函数并进行比较
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = []

for kernel in kernels:
    svm_classifier = SVC(kernel=kernel, C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((kernel, accuracy))
    print(f"\nKernel: {kernel}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # 可视化混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {kernel} Kernel')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# 汇总结果
results_df = pd.DataFrame(results, columns=['Kernel', 'Accuracy'])
print("\n不同核函数的结果比较：")
display(results_df)

# 可视化结果比较
sns.barplot(x='Kernel', y='Accuracy', data=results_df)
plt.title('SVM Kernel Comparison')
plt.ylim(0.8, 1.0)
plt.show()

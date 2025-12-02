"""
完整、清理后的鸢尾花数据集分析脚本。
功能：数据加载、描述性统计、绘图（在 headless 环境下安全）、训练与评估两个模型（LogisticRegression、KNN）、以及性能比较。
"""

import os
import numpy as np
import pandas as pd
import matplotlib

# 检测是否在 IPython / Jupyter 环境中运行
IN_IPYTHON = False
try:
    # get_ipython 存在则认为是在交互式环境（如 Jupyter）
    from IPython import get_ipython
    if get_ipython() is not None:
        IN_IPYTHON = True
except Exception:
    IN_IPYTHON = False

# 在不同环境下选择合适的后端：Notebook -> inline；脚本/CI -> Agg
if IN_IPYTHON:
    try:
        # 在 notebook 中使用内嵌后端（不会触发 FigureCanvasAgg 警告）
        get_ipython().run_line_magic('matplotlib', 'inline')
    except Exception:
        pass
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# 在无图形界面的环境中使用非交互后端，避免 plt.show() 导致阻塞或异常
matplotlib.use('Agg')


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def main():
    iris = load_iris()
    print("数据集键名:", list(iris.keys()))

    # 构建 DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])

    print("数据集前 5 行:")
    print(df.head())
    print("数据集形状:", df.shape)
    print("\n数据集信息:")
    df.info()

    # 描述统计
    print("数值特征的统计描述:")
    print(df.describe())

    # 缺失值检查
    print("缺失值检查:")
    print(df.isnull().sum())

    # 类别分布
    print("类别分布:")
    print(df['species'].value_counts())

    # 尝试设置中文字体（如果系统支持）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    # 绘图：特征直方图
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    df[numeric_cols].hist()
    plt.suptitle("特征分布直方图")
    plt.tight_layout()
    # 在 notebook 中直接显示；否则保存为文件并关闭，避免 FigureCanvasAgg 警告
    if IN_IPYTHON:
        plt.show()
    else:
        os.makedirs('d:\\project\\figures', exist_ok=True)
        plt.savefig(r'd:\\\project\\figures\\features_hist.png', bbox_inches='tight')
        plt.close()

    # pairplot（保护在 headless 环境）
    if IN_IPYTHON:
        try:
            sns.pairplot(df, hue='species', diag_kind='hist')
            plt.suptitle("特征散点图矩阵", y=1.02)
            plt.show()
        except Exception:
            pass
    else:
        try:
            g = sns.pairplot(df, hue='species', diag_kind='hist')
            g.fig.suptitle("特征散点图矩阵", y=1.02)
            os.makedirs('d:\\project\\figures', exist_ok=True)
            g.savefig(r'd:\\\project\\figures\\pairplot.png', bbox_inches='tight')
            plt.close()
        except Exception:
            pass

    # 相关性矩阵（只用数值列）
    numeric_df = df.select_dtypes(include=['number'])
    print("用于计算相关性的数值列:", numeric_df.columns.tolist())
    correlation_matrix = numeric_df.corr()
    print("相关性矩阵:")
    print(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("特征相关性热力图")
    if IN_IPYTHON:
        plt.show()
    else:
        os.makedirs('d:\\project\\figures', exist_ok=True)
        plt.savefig(r'd:\\\project\\figures\\correlation_heatmap.png', bbox_inches='tight')
        plt.close()

    # 特征与标签
    X = iris.data
    y = iris.target
    print("特征矩阵形状:", X.shape)
    print("目标标签形状:", y.shape)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("训练集大小:", X_train.shape)
    print("测试集大小:", X_test.shape)
    print("训练集类别分布:")
    print(pd.Series(y_train).value_counts())
    print("测试集类别分布:")
    print(pd.Series(y_test).value_counts())

    # 逻辑回归（使用默认 multi_class to avoid FutureWarning）
    lr_model = LogisticRegression(solver='lbfgs', random_state=42, max_iter=200)
    lr_model.fit(X_train, y_train)
    y_train_pred_lr = lr_model.predict(X_train)
    y_test_pred_lr = lr_model.predict(X_test)

    train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
    test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
    precision_lr = precision_score(y_test, y_test_pred_lr, average='weighted', zero_division=0)
    recall_lr = recall_score(y_test, y_test_pred_lr, average='weighted', zero_division=0)
    f1_lr = f1_score(y_test, y_test_pred_lr, average='weighted', zero_division=0)

    print("\n逻辑回归模型:")
    print(f"训练集准确率: {train_accuracy_lr:.4f}")
    print(f"测试集准确率: {test_accuracy_lr:.4f}")
    print(f"精确率: {precision_lr:.4f}")
    print(f"召回率: {recall_lr:.4f}")
    print(f"F1分数: {f1_lr:.4f}")

    cm_lr = confusion_matrix(y_test, y_test_pred_lr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("逻辑回归混淆矩阵")
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    if IN_IPYTHON:
        plt.show()
    else:
        os.makedirs('d:\\project\\figures', exist_ok=True)
        plt.savefig(r'd:\\\project\\figures\\lr_confusion.png', bbox_inches='tight')
        plt.close()

    print("逻辑回归分类报告:")
    print(classification_report(y_test, y_test_pred_lr, target_names=iris.target_names))

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_train_pred_knn = knn_model.predict(X_train)
    y_test_pred_knn = knn_model.predict(X_test)

    train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
    test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
    precision_knn = precision_score(y_test, y_test_pred_knn, average='weighted', zero_division=0)
    recall_knn = recall_score(y_test, y_test_pred_knn, average='weighted', zero_division=0)
    f1_knn = f1_score(y_test, y_test_pred_knn, average='weighted', zero_division=0)

    print("\nK近邻模型:")
    print(f"训练集准确率: {train_accuracy_knn:.4f}")
    print(f"测试集准确率: {test_accuracy_knn:.4f}")
    print(f"精确率: {precision_knn:.4f}")
    print(f"召回率: {recall_knn:.4f}")
    print(f"F1分数: {f1_knn:.4f}")

    cm_knn = confusion_matrix(y_test, y_test_pred_knn)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("K近邻混淆矩阵")
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    if IN_IPYTHON:
        plt.show()
    else:
        os.makedirs('d:\\project\\figures', exist_ok=True)
        plt.savefig(r'd:\\\project\\figures\\knn_confusion.png', bbox_inches='tight')
        plt.close()

    print("K近邻分类报告:")
    print(classification_report(y_test, y_test_pred_knn, target_names=iris.target_names))

    # 比较表
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'K-Nearest Neighbors'],
        'Train Accuracy': [train_accuracy_lr, train_accuracy_knn],
        'Test Accuracy': [test_accuracy_lr, test_accuracy_knn],
        'Precision': [precision_lr, precision_knn],
        'Recall': [recall_lr, recall_knn],
        'F1 Score': [f1_lr, f1_knn]
    })

    print("模型性能比较:")
    print(comparison)

    # 绘制比较柱状图（示例）
    try:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(comparison))
        width = 0.35
        plt.bar(x - width/2, comparison['Train Accuracy'], width, label='训练准确率')
        plt.bar(x + width/2, comparison['Test Accuracy'], width, label='测试准确率')
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.title('模型准确率比较')
        plt.xticks(x, comparison['Model'])
        plt.legend()
        plt.ylim(0.0, 1.05)
        plt.tight_layout()
        if IN_IPYTHON:
            plt.show()
        else:
            os.makedirs('d:\\project\\figures', exist_ok=True)
            plt.savefig(r'd:\\\project\\figures\\model_comparison.png', bbox_inches='tight')
            plt.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()



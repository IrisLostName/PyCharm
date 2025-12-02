import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. 数据生成与处理
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += rng.uniform(size=X.shape)

X_ = StandardScaler().fit_transform(X)
data = X_, y

# 2. 模型训练
clf = SVC(kernel="linear", C=1e10, random_state=3)
clf.fit(X_, y)

# 3. 修正后的绘图函数
# 将默认步长 h 从 0.92 改为 0.02
def plot_decision_boundary(data, clf, h=0.02):
    X, y = data

    # 修正坐标轴范围，不要减去太多，留 0.5 的边距即可
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 这里使用 predict 绘制硬边界
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    # 使用 contourf 绘制背景
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)

    # 绘制散点图
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=y, palette=["tab:blue", "tab:orange"], s=50, edgecolor="k"
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Corrected SVM Linear Boundary")
    plt.show()

plot_decision_boundary(data, clf)

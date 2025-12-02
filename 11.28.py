import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X,y=make_circles(
    n_samples=1000,
    noise=0.2, factor=0.2,
    random_state=1
)
plt.figure(figsize=(5,5))
sns.scatterplot(
    x=X[:,0], y=X[:,1],
    hue=y, legend='full'
)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
X_=StandardScaler().fit_transform(X)
data=X_,y

clf=SVC(C=1.0, kernel='rbf', random_state=3)
clf.fit(X_,y)

def plot_predicted_proba(data,clf,h=0.02):
    X,y=data
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdBu, alpha=0.8)
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, legend="full")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()




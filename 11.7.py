from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成示例回归数据并绘制散点图
X, y = datasets.make_regression(
    n_samples=100, n_features=1, n_informative=1, noise=20, random_state=9
)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue')

# 使用正确的参数名：fontsize（而不是 font_size）
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)

# 修正 grid 的参数名为 linestyle（原来的 lineStyles 不存在），使用常见样式
plt.grid(linestyle='-.')

ax = plt.gca()
ax.xaxis.set_label_coords(1.02, 0.04)
ax.yaxis.set_label_coords(-0.04, 1)

# 保存并展示图像，便于在无交互环境下仍能查看结果
plt.tight_layout()
plt.savefig(r"D:\project\regression_scatter.png")
plt.show()

# 下面演示如何切分并标准化数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

lin_r2_score = lin_reg.score(X_test, y_test)
print(f"线性回归模型在测试集上的R²得分: {lin_r2_score:.4f}")


from sklearn.metrics import mean_squared_error
lin_mse=mean_squared_error(y_test, y_pred)
print(f"线性回归模型在测试集上的均方误差: {lin_mse:.4f}")


from sklearn.model_selection import cross_val_predict
predicted=cross_val_predict(lin_reg,X,y,cv=10)
cv_lin_mse=mean_squared_error(y,predicted)
print(f"线性回归模型在全部数据上的10折交叉验证均方误差: {cv_lin_mse:.4f}")

import numpy as np
X=scaler.inverse_transform(X)
X_min=min(X)[0]
X_max=max(X)[0]
X_line=np.linspace(X_min,X_max,100)

y_line=X_line*lin_reg.coef_+lin_reg.intercept_
plt.figure(figsize=(8, 6))
ax=plt.gca()
ax.scatter(X,y,color='blue',label='数据点')
ax.plot(X_line,y_line,color='red',lw=4)
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue')

# 使用正确的参数名：fontsize（而不是 font_size）
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)

# 修正 grid 的参数名为 linestyle（原来的 lineStyles 不存在），使用常见样式
plt.grid(linestyle='-.')

ax = plt.gca()
ax.xaxis.set_label_coords(1.02, 0.04)
ax.yaxis.set_label_coords(-0.04, 1)

plt.tight_layout()
plt.savefig(r"D:\project\123456789.png")
plt.show()


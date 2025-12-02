import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 加载和预处理数据
raw = pd.read_csv("D:\project\DATASET-B.csv")
feature = raw[(raw.rowid < 30) & (raw.colid > 20) & (raw.date == 20161101)]
s1 = feature[feature.time_id == 47].set_index(['rowid', 'colid']).aveSpeed
s2 = feature[feature.time_id == 48].set_index(['rowid', 'colid']).aveSpeed
s3 = feature[feature.time_id == 49].set_index(['rowid', 'colid']).aveSpeed
s4 = feature[feature.time_id == 50].set_index(['rowid', 'colid']).aveSpeed
data = pd.DataFrame(pd.concat((s1, s2, s3, s4), axis=1).values).dropna().reset_index(drop=True)
data.columns = ["47", "48", "49", "50"]

# 1. 定义特征和目标变量
X = data[["47", "48", "49"]].values
y = data["50"].values.reshape(-1,1)

reg=LinearRegression()
reg.fit(X,y)
coef=reg.coef_
cons=reg.intercept_[0]
print("变量参数：",coef,"常数项：",cons,"R²",reg.score(X,y))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(8, 6))
plt.scatter(reg.predict(X), y, color='blue')
plt.plot([3,18],[3,18],'--',color='red',lw=4)
plt.xlim(3,18)
plt.ylim(3,18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(linestyle='-.')
plt.xlabel('预测值', fontsize=20)
plt.ylabel('真实值', fontsize=20)
plt.show()

from sklearn.metrics import r2_score, mean_squared_error
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
print(f"模型的 R² (决定系数) 为: {r2:.4f}")
mse = mean_squared_error(y, y_pred)
print(f"模型的均方误差 (MSE) 为: {mse:.4f}")

from sklearn.model_selection import cross_val_predict
predicted=cross_val_predict(lin_reg,X,y,cv=10)
cv_lin_mse=mean_squared_error(y,predicted)
print(f"线性回归模型在全部数据上的10折交叉验证均方误差: {cv_lin_mse:.4f}")






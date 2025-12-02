import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
print(f"(3) 模型在测试集上的 R2 系数: {r2:.4f}")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n(4) 模型评价与验证:")
print(f"    平均绝对误差 (MAE): {mae:.4f}")
print(f"    均方误差 (MSE): {mse:.4f}")
print(f"    均方根误差 (RMSE): {rmse:.4f}")

# 模型评价解读
if r2 > 0.7:
    print("    模型评价: 模型拟合效果良好。")
elif r2 > 0.4:
    print("    模型评价: 模型拟合效果一般。")
else:
    print("    模型评价: 模型拟合效果较差。")


X_scaled = scaler.fit_transform(X)
cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='r2')

print("\n(5) 10折交叉验证:")
print(f"    每次交叉验证的 R2 分数: {np.round(cv_scores, 4)}")
print(f"    交叉验证 R2 分数的平均值: {cv_scores.mean():.4f}")
print(f"    交叉验证 R2 分数的标准差: {cv_scores.std():.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='训练数据', alpha=0.7)
plt.scatter(X_test, y_test, color='green', label='测试数据', alpha=0.7)

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_range_scaled = scaler.transform(x_range)
y_range_pred = model.predict(x_range_scaled)
plt.plot(x_range, y_range_pred, color='red', linewidth=2, label='线性回归模型')

plt.title('线性回归模型可视化')
plt.xlabel('特征 (Feature)')
plt.ylabel('目标值 (Target)')
plt.legend()
plt.grid(True)
plt.show()

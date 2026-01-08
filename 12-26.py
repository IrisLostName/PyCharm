import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("DATASET-B.csv")
int_cols = ["rowid", "colid", "time_id"]
df[int_cols] = df[int_cols].astype(int)
df = df.sort_values(by=["time_id", "rowid"]).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["weekday"] = df["date"].dt.weekday
df_sampled = df.sample(n=50000, random_state=233).reset_index(drop=True)

X = df_sampled.drop(["labels", "date"], axis=1)
y = df_sampled["labels"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

gbdt = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=256,
    max_depth=8,
    subsample=0.8,
    max_features=0.9,
    min_samples_split=5,
    min_samples_leaf=30,
    random_state=233,
)

gbdt.fit(X_train, y_train)

y_pred = gbdt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率 (Accuracy): {accuracy:.4f}")
print("\n分类报告:\n", classification_report(y_test, y_pred))

plot_df = X_test.copy()
plot_df["pred_labels"] = y_pred


sns.set(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=plot_df,
    x="aveSpeed",
    y="stopNum",
    hue="pred_labels",
    palette="viridis",
    s=60,
    alpha=0.6,
)

plt.title("GBDT 分类结果可视化 (测试集预测)", fontsize=15)
plt.xlabel("平均速度 (aveSpeed)", fontsize=12)
plt.ylabel("停靠次数 (stopNum)", fontsize=12)
plt.legend(title="预测标签")
plt.savefig("gbdt_classification_result.png")
plt.show()

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(gbdt.feature_importances_, index=X.columns)
feat_importances.nlargest(10).sort_values().plot(kind="barh", color="skyblue")
plt.title("GBDT 特征重要性排名", fontsize=15)
plt.xlabel("重要性得分")
plt.savefig("gbdt_feature_importance.png")
plt.show()

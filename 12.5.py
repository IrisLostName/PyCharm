import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    '平均速度': [6.25, 5.70, 12.15, 6.61, 16.02, 5.52, 15.49, 9.48, 7.16,
                 10.56, 11.47, 3.98, 14.71, 16.02, 11.11, 15.37, 12.03, 6.19],
    '流量': [5, 8, 2, 3, 16, 3, 21, 15, 21,
             10, 12, 7, 21, 21, 5, 4, 15, 6],
    # '有无停车' 列并未作为特征要求，故此处不使用
    '交通状态': ['拥堵', '拥堵', '畅通', '缓行', '畅通', '缓行', '畅通', '拥堵', '拥堵',
                 '缓行', '畅通', '拥堵', '畅通', '畅通', '缓行', '畅通', '畅通', '拥堵']
}

df = pd.DataFrame(data)
feature_names = ['平均速度', '流量']
X = df[feature_names]
y = df['交通状态']

clf = tree.DecisionTreeClassifier(criterion='gini', random_state=42)
clf = clf.fit(X, y)

plt.figure(figsize=(12, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

tree.plot_tree(clf,
               feature_names=feature_names,
               class_names=clf.classes_,
               filled=True,
               rounded=True,
               fontsize=10)

plt.title("交通状态分类决策树")
plt.show()

text_representation = tree.export_text(clf, feature_names=feature_names)
print("\n=== 决策树规则文本版 ===\n")
print(text_representation)


test_sample = pd.DataFrame([[5.0, 20]], columns=['平均速度', '流量'])
prediction = clf.predict(test_sample)
print(f"\n测试预测 (速度=5.0, 流量=20): 预测状态为 -> {prediction[0]}")

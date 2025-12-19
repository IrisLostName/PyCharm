import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


data_ori=pd.read_csv("聚类数据集.csv")
feature=['stopNum','aveSpeed']
scaler=StandardScaler()
scaler.fit(data_ori[feature])
data_ori_nor=scaler.transform(data_ori[feature])

n=3
GMM=GaussianMixture(n,random_state=0).fit(data_ori_nor)
labels=GMM.predict(data_ori_nor)
num_iter=GMM.n_iter_
print(num_iter)  # 11

output_data=pd.concat(
    (
        data_ori,pd.DataFrame(labels,columns=["labels"])
    ),axis=1
)

output_data.to_csv('GMM聚类结果.csv',index=False)

df = pd.read_csv('GMM聚类结果.csv')
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.scatterplot(data=df, x='aveSpeed', y='stopNum', hue='labels', palette='viridis', s=100, alpha=0.7)
plt.title('GMM聚类结果', fontsize=15)
plt.xlabel('平均速度', fontsize=12)
plt.ylabel('停靠站数量', fontsize=12)
plt.legend(title='聚类标签')
plt.savefig('GMM聚类可视化.png')




import pandas as pd
import numpy as np
from docutils.nodes import label
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy.core.random import sample
from sympy.strategies.branch import condition

data=pd.read_csv("DATASET-B.csv",nrows=20000)

x = data[['aveSpeed', 'stopNum', 'volume', 'speed_std']]
y=pd.Categorical(data['labels']).codes
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=1)

model=DecisionTreeClassifier(
    criterion='gini',
    min_samples_split=10,
    min_samples_leaf=40,
    max_depth=10,
    class_weight='balanced'
)
model.fit(x_train,y_train)
y_test_hat=model.predict(x_test)
y_test=y_test.reshape(-1)
result=(y_test_hat==y_test)
acc=np.mean(result)
print('准确率：%.2f%%' % (acc*100))

print(model.feature_importances_)

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
N,M=50,50
x1_min,x2_min,_,_=x.min()
x1_max,x2_max,_,_=x.max()
t1=np.linspace(x1_min,x1_max,N)
t2=np.linspace(x2_min,x2_max,M)
x1,x2=np.meshgrid(t1,t2)
x_show=np.stack((x1.flat,x2.flat),axis=1)
model2=DecisionTreeClassifier(
    criterion='gini',
    min_samples_split=10,
    min_samples_leaf=40,
    max_depth=10,
    class_weight='balanced'
)
model2.fit(x_train[['aveSpeed','stopNum']],y_train)
y_show_hat=model2.predict(x_show)
y_show_hat=y_show_hat.reshape(x1.shape)
sample_plot_idx=np.random.choice(x_test.shape[0],200,replace=False)
x_test1=x_test.iloc[sample_plot_idx]
y_test1=y_test[sample_plot_idx]
plt.figure(facecolor='w',dpi=300)
plt.pcolormesh(x1,x2,y_show_hat,alpha=0.1)
plt.show()
condition=['畅通','缓行','拥堵']
color=['purple','green','yellow']
for i in range(3):
    plot_idx=np.where(y_test1==i)
    plt.scatter(
        x_test1.iloc[plot_idx]['aveSpeed'],
        x_test1.iloc[plot_idx]['stopNum'],
        c=color[i],edgecolors='k',s=20,zorder=10,label=condition[i]
    )
plt.xlabel('停车次数(次)')
plt.ylabel('平均车速(km/h)')
plt.title('基于决策树的交通状态分类',fontsize=15)
plt.legend()
plt.grid()
plt.show()

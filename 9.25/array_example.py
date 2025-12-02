import numpy as np

# 创建 3x3 的二维数组
a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

print(a>2)

print(a[a>2])

print(a[(a>2) & (a<6)])
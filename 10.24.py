import pandas as pd
from scipy.interpolate import lagrange
import numpy as np

traj=pd.read_csv("D:\project\DATASET-A.csv",header=None,usecols=[2,3,4]).iloc[1:16]
traj.columns=["timestamp","lon","lat"]


# 将列转换为数值类型以便计算
for col in ['timestamp', 'lon', 'lat']:
    traj[col] = pd.to_numeric(traj[col], errors='coerce')

print("----------- 原始数据 -----------")
print(traj)

# 1. 使用 .diff() 找到时间间隔大于等于6秒的行
time_gaps = traj['timestamp'].diff()
gap_indices = time_gaps[time_gaps >= 6].index

# 2. 准备一个列表来收集所有需要新插入的行
new_rows = []
k = 3 # 在缺失点前后各取k个点进行插值，k值可调

for i in gap_indices:
    # 定义插值所需的数据范围
    before_gap = traj.iloc[max(0, i - k) : i]
    after_gap = traj.iloc[i : min(len(traj), i + k)]
    interp_data = pd.concat([before_gap, after_gap])

    # 创建拉格朗日插值函数
    poly_lon = lagrange(interp_data['timestamp'].values, interp_data['lon'].values)
    poly_lat = lagrange(interp_data['timestamp'].values, interp_data['lat'].values)

    # 计算需要插入的缺失点
    start_time = traj.loc[i - 1, 'timestamp']
    new_time = start_time + 3  # 在此场景下只缺失一个点

    new_lon = poly_lon(new_time)
    new_lat = poly_lat(new_time)

    new_rows.append({
        'timestamp': new_time,
        'lon': f"{new_lon:.7f}",
        'lat': f"{new_lat:.8f}",
        'modified': 'Yes' # 标记为已修改
    })

# 3. 将原始数据和新行数据合并
if new_rows:
    traj['modified'] = 'No' # 标记原始数据
    interpolated_df = pd.DataFrame(new_rows)
    traj = pd.concat([traj, interpolated_df], ignore_index=True)
    traj = traj.sort_values(by='timestamp').reset_index(drop=True)

print("\n----------- 插值后数据 -----------")
print(traj)

print("\n----------- 被插补的行 -----------")
print(traj[traj['modified'] == 'Yes'])

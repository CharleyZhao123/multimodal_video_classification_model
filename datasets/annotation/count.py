import os
import numpy as np
import pandas as pd

res = np.zeros((5,2,5))
label_map = {'end_action':0,'lchange':1,'lturn':2,'rchange':3,'rturn':4}
for i in range(5):
    csv_path = './fold'+str(i)+'_ori.csv'
    data = pd.read_csv(csv_path,header=None)
    data_list = data.values.tolist()
    for video_info in data_list:
        if video_info[3]=='training':
            res[i,0,label_map[video_info[1]]] += 1
        elif video_info[3]=='validation':
            res[i,1,label_map[video_info[1]]] += 1
print(res)
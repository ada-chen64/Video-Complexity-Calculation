import numpy as np
import pandas as pd


data = np.load('train_data.npy')
# print(data.shape)

norm1 = data[:, 1:15]
# print(norm1.shape)
norm2 = data[:, 15:]
# print(norm2.shape)

max1 = np.max(norm1, 0)
min1 = np.min(norm1, 0)
# print(max1)
# print(min1)
max2 = np.max(norm2).repeat(6)
min2 = np.min(norm2).repeat(6)
# print(max2)
# print(min2)

max0 = np.hstack((max1, max2))
min0 = np.hstack((min1, min2))
norm = np.vstack((max0, min0))
# print(norm)
np.save('data/norm.npy', norm)

# def normalize_data(matrix, axis=None):
#     return matrix


# print('load data')
# file = pd.read_csv("labels.csv")
# data = file.iloc[:536,[1]].values
# file = pd.read_csv('lap.csv')
# lap_feat = file.iloc[:536, 1:11].values
# bit_feat = file.iloc[:536, [11]].values
# ssim_feat = file.iloc[:536, [12]].values
# file = pd.read_csv('optical_flow.csv')
# of_feat = file.iloc[:536, [1]].values
# file = pd.read_csv('sobel_space.csv')
# sp_feat = file.iloc[:536, [1]].values
# file = pd.read_csv('sobel_time.csv')
# st_feat = file.iloc[:536, 1:].values


# lap_norm = normalize_data(lap_feat, 0)
# bit_norm = normalize_data(bit_feat)
# ssim_norm = normalize_data(ssim_feat)
# of_norm = normalize_data(of_feat)
# sp_norm = normalize_data(sp_feat)
# st_norm = normalize_data(st_feat)


# data = np.hstack((data, lap_norm))
# data = np.hstack((data, bit_norm))
# data = np.hstack((data, ssim_norm))
# data = np.hstack((data, of_norm))
# data = np.hstack((data, sp_norm))
# data = np.hstack((data, st_norm))


# print(data.shape)
# # np.savetxt('data/train_data.txt', data, fmt='%.2f')
# np.save('data/train_data.npy', data)
import os
import cv2 
import pandas as pd
import numpy as np
import openpyxl
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def normalizedata(matrix):
    maxnum = np.max(matrix)
    minnum = np.min(matrix)
    diff = maxnum - minnum
    newmatrix = matrix
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            newmatrix[r][c] = (matrix[r][c] - minnum) / diff
    return newmatrix

def Sobel_space(vidpath, clipname):
    sample = 10
    SI_mean_total = 0
    cap = cv2.VideoCapture(vidpath)
    wid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)     # 3: frame width
    hei = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    # 4: frame height
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)     # 7: frame count

    # uniformly and randomly sample
    # eg. sample 10 frames from 50 frames
    #     sample 0 from frame 0 ~ frame 4 randomly
    #     sample 1 from frame 5 ~ frame 9 randomly
    interval = int(cnt // sample)
    indices = np.array(range(sample)) * interval    \
            + np.random.randint(0, interval, sample)
    
    # read the frames
    kps = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        dimensions = img.shape
    
        # height, width, number of channels in image
        
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        gradient = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
        mean = np.mean(gradient)
        #写入新excel    
        SI_mean_total += mean
    feat = [SI_mean_total/sample]
    return feat
def Sobel_time(vidpath, clipname):
    feat = []
    sample = 6     # sample frames' number
    duration = 5    # 6 continuous frames 
    
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)     # 3: frame width
    hei = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    # 4: frame height
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)     # 7: frame count

    interval = int(cnt // sample)
    indices = np.array(range(sample)) * interval    \
            + np.random.randint(0, interval - duration, sample)
    
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read() #prev
        img = cv2.GaussianBlur(frame, (3,3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        
        gradient_prev = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
        SI_total = 0
        for d in range(1, duration):
            ret, frame = cap.read()
            img = cv2.GaussianBlur(frame, (3,3), 0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
            sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

            abs_sobelx = cv2.convertScaleAbs(sobelx)
            abs_sobely = cv2.convertScaleAbs(sobely)
            
            
            gradient = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
            
            diff = abs(gradient - gradient_prev) 
            gradient_prev = gradient 
            SI_mean = np.mean(diff* fps) 
            SI_total += SI_mean
            
        feat.append(SI_total / 4)
    return feat   

#open train features
#labels
# file = pd.read_csv("labels.csv")
# data = file.iloc[:,[0]].values
#laplace (10 columns)
file = pd.read_csv("lap.csv")
lap_feat = file.iloc[:,1:11].values
bit_feat = file.iloc[:, [11]].values
ssim_feat = file.iloc[:, [12]].values
#bitrate (1 column)
# file = pd.read_csv("bitrate.csv")
# bit_feat = file.iloc[:,:].values
# #ssim (1 column)
# file = pd.read_csv("ssim.csv")
# ssim_feat = file.iloc[:,:].values
#optical flow (10 columns or 1 column)
file = pd.read_csv("optical_flow.csv")
of_feat = file.iloc[:,12:22].values
# Sobel space (1 column)
file = pd.read_csv("Sobel_SI2.csv")
sp_feat = file.iloc[:,[14]].values
# Sobel time (6 columns)
file = pd.read_csv("Sobel_time.csv")
st_feat = file.iloc[:,1:7].values
print(sp_feat)
print("new")
#collect features for new videos
path = "C:\\Users\\baobe\\Videos\\test_clips\\clips\\"
listfiles = os.listdir(path)

for file in listfiles:
    if file.endswith(".mp4"):
        clipname = file.split(".")[0]
        #sobel space
        sp_feat = np.vstack((sp_feat, Sobel_space(path +file, clipname)))
        st_Feat = np.vstack((st_feat, Sobel_time(path + file, clipname)))

print(sp_feat)
# normalize data

#laplace
lp = lap_feat[:, [1]]
lap_norm = normalizedata(lp)
for i in range(2,11):
    lp = data[:, [i]]
    # print("ssim pre normalize")
    #print(of)
    lpdata= normalizedata(lp)
    lap_norm=np.hstack((lap_norm, lpdata))
#bitrate and ssim
bitrate_norm = normalizedata(bit_feat)
ssim_norm = normalizedata(ssim_feat)
#optical flow
of_norm = normalizedata(of_feat)
#sobel
sp_norm = normalizedata(sp_feat)
st_norm = normalizedata(st_feat)

data = np.hstacl((data,lap_norm))
data = np.hstack((data, bitrate_norm))
data = np.hstack((data, ssim_norm))
data = np.hstack((data, of_norm))
data = np.hstack((data, sp_norm))
data = np.hstack((data, st_norm))

#train and predict


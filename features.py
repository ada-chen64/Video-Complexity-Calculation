import os
import cv2
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def normalize_data(matrix, axis=None):
    ''' 
    normalize each feature matrix respectively

    Args:
        axis: if None, normalize the whole matrix
              if zero, normalize columns respectively
    '''
    matrix = np.array(matrix)
    maxnum = np.max(matrix, axis)
    minnum = np.min(matrix, axis)
    return (matrix - minnum) / (maxnum - minnum)


def sobel_space(vidpath, clipname):
    cap = cv2.VideoCapture(vidpath)
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # uniformly and randomly sample
    sample = 10    
    interval = int(cnt // sample)
    indices = np.array(range(sample)) * interval    \
            + np.random.randint(0, interval, sample)
    
    # calculate sobel space
    SI_mean_total = 0
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        sobelx = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=-1)
        sobely = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=-1)
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        gradient = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
        mean = np.mean(gradient)
        SI_mean_total += mean

    feat = [SI_mean_total/sample]
    return feat


def sobel_time(vidpath, clipname):
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # uniformly and randomly sample
    sample = 6
    duration = 5
    interval = int(cnt // sample)
    indices = np.array(range(sample)) * interval    \
            + np.random.randint(0, interval - duration, sample)
    
    # calculate sobel time
    feat = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()     # prev
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
            SI_mean = np.mean(diff * fps)
            SI_total += SI_mean
            
        feat.append(SI_total / 4)
    return feat   

class Flow:
    lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    
 
    feature_params = dict( maxCorners = 300, 
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    def __init__(self, video_src):
        self.track_len = 10
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frames = self.cam.get(7)
        self.frame_idx = 0
        self.detect_interval = int(self.frames / 6)
 
    def run(self):
        out = []
        flag = 0
        while True:
            ret, frame = self.cam.read()
            if ret == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()
    
                if len(self.tracks) > 0 and flag < 5:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    d.sort()
                    out.append(sum(d[-5:]) / 5)
                    flag += 1
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
    
                if self.frame_idx % self.detect_interval == 0:
                    flag = 0
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])
    
                self.frame_idx += 1
                self.prev_gray = frame_gray
            
            else:
                return out

'''
open train features
'''
print('load data')

# labels
file = pd.read_csv("labels.csv")
data = file.iloc[:,[0]].values
# # laplace (10 columns)
# file = pd.read_csv('lap.csv')
# lap_feat = file.iloc[:,1:11].values
# bit_feat = file.iloc[:, [11]].values
# ssim_feat = file.iloc[:, [12]].values
# # bitrate (1 column)
# file = pd.read_csv("bitrate.csv")
# bit_feat = file.iloc[:,:].values
# # ssim (1 column)
# file = pd.read_csv("ssim.csv")
# ssim_feat = file.iloc[:,:].values
# # optical flow (10 columns or 1 column)
# file = pd.read_csv('optical_flow.csv')
# of_feat = file.iloc[:,12:22].values
# Sobel space (1 column)
file = pd.read_csv('sobel_space.csv')
sp_feat = file.iloc[:, [1]].values
# Sobel time (6 columns)
file = pd.read_csv('sobel_time.csv')
st_feat = file.iloc[:, 1:7].values


'''
collect features
'''
print('load clips')

path = './clips/'
listfiles = os.listdir(path)
for file in listfiles:
    if file.endswith(".mp4"):
        clipname = file.split(".")[0]
        sp_feat = np.vstack((sp_feat, sobel_space(path + file, clipname)))
        st_feat = np.vstack((st_feat, sobel_time(path + file, clipname)))
        of_feat = np.vstack((of_feat, Flow(path + file, clipname).run()))


'''
normalize data
'''
# # laplace
# lap_nrom = normalize_data(lap_feat, 0)
# # bitrate and ssim
# bit_norm = normalize_data(bit_feat)
# ssim_norm = normalize_data(ssim_feat)
# # optical flow
# of_norm = normalize_data(of_feat)
# sobel
sp_norm = normalize_data(sp_feat)
st_norm = normalize_data(st_feat)

# concatenate data
# data = np.hstack((data, lap_norm))
# data = np.hstack((data, bit_norm))
# data = np.hstack((data, ssim_norm))
# data = np.hstack((data, of_norm))
data = np.hstack((data, sp_norm))
data = np.hstack((data, st_norm))


'''
predict results
'''

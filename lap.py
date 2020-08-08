import numpy as np
import pandas as pd
import cv2
import os, sys, json
import itertools

def clip_and_resize(mat):
    height, width, channel = mat.shape
    cheight = width / 2.390625
    offset = int((height - cheight) // 2)
    ret = cv2.resize(mat[offset:-offset], (1280, int(1280//2.390625)))
    # print(ret.shape)
    return ret

def lap88(mat):
    res = 0
    for i in range(mat.shape[0] // 16):
        for j in range(mat.shape[1] // 16):
            frame = mat[16*i+1:16*(i+1)-1, 16*j+1:16*(j+1)-1]
            lap = cv2.Laplacian(frame, cv2.CV_8U, ksize = ksize, borderType = cv2.BORDER_ISOLATED) + cv2.Laplacian(-frame, cv2.CV_8U, ksize = ksize, borderType = cv2.BORDER_ISOLATED)
            res += lap.sum()
    return res

def lapKer(mat):
    frame = mat # cv2.cvtColor(mat, cv2.COLOR_RGB2YUV)
    imgkernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    sum1 = cv2.filter2D(frame[:,:,0],cv2.CV_32F,imgkernel)**2
    sum2 = cv2.filter2D(frame[:,:,1],cv2.CV_32F,imgkernel)**2
    sum3 = cv2.filter2D(frame[:,:,2],cv2.CV_32F,imgkernel)**2
    return sum1.sum()+sum2.sum()+sum3.sum()

def lapHsv(mat):
    frame = cv2.cvtColor(mat, cv2.COLOR_RGB2HSV)
    lap = cv2.Laplacian(frame, -1) + cv2.Laplacian(-frame, -1)
    return lap

def mosaic(mat):
    h1, h2, _ = mat.shape
    res = np.zeros(mat.shape, dtype = np.float32)
    for i in itertools.count(0, 4):
        if i >= h1: break
        for j in itertools.count(0, 4):
            if j >= h2: break
            for k in range(3):
                tmp = mat[i:i+4,j:j+4,k].sum()/16
                res[i:i+4,j:j+4,k] = tmp
    return res

def demosaicing(cap):
    res = []
    frames = getnframe(cap, 10)
    for frame in frames:

        f2 = mosaic(frame)
        tmp = ((frame.astype(np.uint32)-f2.astype(np.uint32))**2).sum()

        res.append(float(tmp**.5))
    return res


def getnframe(cap, num): #从视频中平均取num帧
    framecnt = cap.get(7) #获取帧数
    group = int(framecnt // num)
    res = []
    while True:
        if len(res) == num : break
        ret, frame = cap.read()
        if not ret: break
        res.append(frame)
        for i in range(group-1):
            ret, frame = cap.read()
            if not ret: break
    return res


# def lap(cap):
#     res = []
#     frames = getnframe(cap, 10)
#     for frame in frames:
#         # frame = clip_and_resize(frame)

#         lapf = cv2.cv2.Laplacian(frame, -1) + cv2.Laplacian(-frame, -1)
#         lapf = lapf.astype(np.float32)
#         # lapf = lapf * lapf
#         res.append(float(lapf.std()))
#         # res.append(lapf[:,:,0].std()+lapf[:,:,1].std()+lapf[:,:,2].std())
#         # res.append(lapKer(frame))

#     # return pd.Series(res).mean()
#     return res

def lap(cap):
    res1 = []
    res2 = []
    while True:
        ret, frame = cap.read()
        if not ret: break

        lapf = cv2.Laplacian(frame, -1) + cv2.Laplacian(-frame, -1)
        lapf = lapf.astype(np.float32)
        lapf = lapf * lapf
        res1.append(float(lapf.mean()**.5))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        lapf = cv2.Laplacian(frame, -1) + cv2.Laplacian(-frame, -1)
        lapf = lapf.astype(np.float32)
        res2.append(float(lapf.std()))

    res1 = pd.Series(res1)
    res2 = pd.Series(res2)
    return res1.mean(), res1.std(ddof=0), res1.max(), res1.median(), res1.min(), \
           res2.mean(), res2.std(ddof=0), res2.max(), res2.median(), res2.min()

def fromfile(fullpath, func):
    cap = cv2.VideoCapture(fullpath)
    return func(cap)

def dftdiff(f1, f2):
    res = 0
    bs = 16
    w, h = f1.shape
    for i in range(w//bs):
        for j in range(h//bs):
            fr1 = f1[bs*i:bs*(i+1), bs*j:bs*(j+1)]
            fr2 = f2[bs*i:bs*(i+1), bs*j:bs*(j+1)]
            ft1 = cv2.dft(fr1, flags = cv2.DFT_COMPLEX_OUTPUT)
            ft2 = cv2.dft(fr2, flags = cv2.DFT_COMPLEX_OUTPUT)
            ft1 = ft1[:8,:8]
            ft2 = ft2[:8,:8]
            ft1r = (ft1**2).sum(-1)**.5
            ft2r = (ft2**2).sum(-1)**.5
            resp = ((ft1r-ft2r)**2).sum()**.5
            res += resp
    return res

def takeframe(cap, num):
    res = []
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        res.append(frame)
        
        for i in range(num-1):
            ret, frame = cap.read()
    return res

def dftdff(cap):
    res = []
    frames = takeframe(cap, 5)
    for prev, frame in zip(itertools.islice(frames, 1, None), frames):
        frame_sc = dftdiff(prev.astype(np.float32), frame.astype(np.float32))
        res.append(frame_sc)

    res = pd.Series(res)
    return res.mean(), res.std(ddof=0), res.max(), res.median(), res.min()

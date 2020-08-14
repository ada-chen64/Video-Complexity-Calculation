import re
import cv2
import numpy as np
import pandas as pd


def normalize_data(data, norm):
    '''
    normalize data from [min, max] to [0, 1]

    Param:
        norm: [[max0, max1, ..., max19], [min0, min1, ..., min19]]
        data: [[lap 0-9], [bitrate 10], [ssim 11], [optical 12], [sobel space 13], [sobel time 14-19]]
    Return:
        data: (data - min) / (max - min)
    '''

    return (data - norm[1]) / (norm[0] - norm[1])


def extract_feature(base):
    '''
    collect features from all test clips

    Param:
        clips: clips' names, eg. 'clips_1.mp4'
    Return:
        data: [[lap 0-9], [bitrate 10], [ssim 11], [optical 12], [sobel space 13], [sobel time 14-19]]
    '''
    
    clip_path = '360p/clips/' + base + '.mp4'
    info_path = '360p/output/' + base + '.txt'

    data = [lap_space(clip_path) + bitrate_ssim(info_path) + optical_flow(clip_path)
            + sobel_space(clip_path) + sobel_time(clip_path)]
        
    return np.array(data)


def sobel_space(path):
    cap = cv2.VideoCapture(path)
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


def sobel_time(path):
    cap = cv2.VideoCapture(path)
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


def optical_flow(path):
    lk_params = dict(winSize  = (15, 15), maxLevel = 2, 
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    
    feature_params = dict(maxCorners = 300, qualityLevel = 0.3,
                          minDistance = 7, blockSize = 7)
    track_len = 10
    tracks = []

    cap = cv2.VideoCapture(path)
    frames = cap.get(7)
    frame_idx = 0
    detect_interval = int(frames / 10)

    out = []
    flag = 0
    while True:
        ret, frame = cap.read()
        if ret == True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
    
            if len(tracks) > 0 and flag < 3:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                d.sort()
                out.append(sum(d[-5:]) / 5)
                flag += 1
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                tracks = new_tracks
    
            if frame_idx % detect_interval == 0:
                flag = 0
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])
    
            frame_idx += 1
            prev_gray = frame_gray
            
        else:
            start = int((len(out) - 10) / 2)
            return [np.mean(out[start:start + 10])]


def bitrate_ssim(path):
    p1 = re.compile(" ([0-9\.]*?) kb/s")
    p2 = re.compile("SSIM Mean Y: (.*?) ")

    file = open(path, mode='r')
    text = file.readlines()[-1]
    bitrate = float(p1.search(text).group(1))
    ssim = float(p2.search(text).group(1))
    file.close()

    return [bitrate, ssim]


def lap_space(path):
    cap = cv2.VideoCapture(path)
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
    return [res1.mean(), res1.std(ddof=0), res1.max(), res1.median(), res1.min(), \
            res2.mean(), res2.std(ddof=0), res2.max(), res2.median(), res2.min()]

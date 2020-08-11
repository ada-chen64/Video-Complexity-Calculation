import time
import numpy as np

from clip import cut_clips
from feat import extract_features, normalize_data
from pred import predict_complexity


# step0: get clips' names
t0 = time.time()
file = './clips.txt'
info = np.loadtxt(file, dtype=np.str, delimiter=',')
clips = info # TODO: input file format
# clips = info[:, 0]


# step1: cut clips to 360p
print('clips ...')
t1 = time.time()
cut_clips(clips)
print('time: %.2fs' % (time.time() - t1))
print()


# step2: extract features
print('features ...')
t2 = time.time()
data = extract_features(clips) # TODO: sobel time rank/ optical 10/1
print('time: %.2fs' % (time.time() - t2))
print()


# step3: normalize data
print('normalize ...')
t3 = time.time()
norm = np.load('data/norm.npy')
data = normalize_data(data, norm)
print('time: %.2fs' % (time.time() - t3))
print()


# step4: calculate complexity
print('complexity ...')
t4 = time.time()
preds = predict_complexity(data)
print('time: %.2fs' % (time.time() - t4))
print()


# step5: output preds
# TODO: output files of preds
print(preds)
print('all time: %.2fs' % (time.time() - t0))
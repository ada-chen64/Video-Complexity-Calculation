import time
import numpy as np

from clip import cut_clip
from feat import extract_feature, normalize_data, bitrate_ssim
from pred import predict_complexity


# step0: get clips' path and load training data
file = './input.txt'
clips = np.loadtxt(file, dtype=np.str, delimiter=',')
norm = np.load('data/norm.npy')


for clip in clips:

    t0 = time.time()

    # step1: cut clips to 360
    base = cut_clip(clip)

    # step2: extract features
    data = extract_feature(base)

    # step3: normalize data
    data = normalize_data(data, norm)

    # step4: calculate complexity
    pred = predict_complexity(data)

    t = time.time() - t0

    # step5: cut according to the pred
    bitrates = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]
    cut_clip(clip, bitrates[pred])
    bitrate, ssim = bitrate_ssim('720p/output/' + base + '.txt')

    # step6: output
    print('%s, %d, %.4fs, %.2fkb/s, %.6f' % (clip, pred.item(), t, bitrate, ssim))

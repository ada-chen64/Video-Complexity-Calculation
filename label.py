import os
import re
import numpy as np
from pred import performance


file = './input.txt'
clips = np.loadtxt(file, dtype=np.str, delimiter=',')
bitrates = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]

if not os.path.exists('labels'):
    os.mkdir('labels')
if not os.path.exists('labels/clips/'):
    os.mkdir('labels/clips')
if not os.path.exists('labels/output/'):
    os.mkdir('labels/output/')


labels = []
for clip in clips:
    path, ext = os.path.splitext(clip)
    base = os.path.basename(path)

    ssims = []
    for bitrate in bitrates:
        src = clip
        tar = 'labels/clips/%s_%d.mp4' % (base, bitrate)
        out = 'labels/clips/%s_%d.txt' % (base, bitrate)
        bufsize = 2 * bitrate
        maxrate = 1.5 * bitrate
        cmd = 'ffmpeg -i %s -vf scale=1280x720 -an -c:v libx265 -tune ssim -tag:v hvc1  \
            -colorspace bt709 -color_primaries bt709 -pix_fmt yuv420p -aspect 16:9 -vsync 0 \
            -x265-params ssim=1:keyint=50:min-keyint=50:vbv-bufsize=%d:vbv-maxrate=%d:scenecut=0:scenecut-bias=0:open-gop=0 \
            -b:v %dk -y %s 2> %s' % (src, bufsize, maxrate, bitrate, tar, out)
        
        if not os.path.exists(tar):
            os.system(cmd)
        print(tar)
        
        file = open(out)
        lines = file.readlines()
        match = re.search('SSIM Mean Y: (.*?) .*', lines[-2] + lines[-1])
        ssim = float(match.group(1))
        ssims.append(ssim)
        file.close()
    
    print(ssims)
    for i, ssim in enumerate(ssims):
        if ssim > 0.970:
            labels.append(i)
            break
    else:
        labels.append(9)


print(labels)
preds = np.loadtxt('output.txt', dtype=np.str, delimiter=',')
preds = preds[1].astype(int)
performance(preds, labels, clips)

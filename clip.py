import os
import time
import numpy


def cut_clips(clips):
    '''
    360p clips for features and information extraction
    '''

    # make dictionary
    if not os.path.exists('360p'):
        os.mkdir('360p')
    if not os.path.exists('360p/clips'):
        os.mkdir('360p/clips')
    if not os.path.exists('360p/output'):
        os.mkdir('360p/output')

    # ffmpeg crf mode
    for clip in clips:
        src = 'clips/' + clip
        tar = '360p/clips/' + clip
        out = '360p/output/' + ''.join(clip.split('.')[:-1]) + '.txt'
        if os.path.exists(tar):
            continue
        cmd = 'ffmpeg -i %s -vf scale=640x360 -an -c:v libx265 -tune ssim -tag:v hvc1 \
            -colorspace bt709 -color_primaries bt709 -pix_fmt yuv420p -aspect 16:9 -vsync 0         \
            -x265-params ssim=1:keyint=50:min-keyint=50:vbv-bufsize=2200:vbv-maxrate=1650:scenecut=0:scenecut-bias=0:open-gop=0\
            -crf 26 -y %s 2> %s' % (src, tar, out)
        try:
            os.system(cmd)
        except:
            print('clip error:', end=' ')
            print(tar)

def cut_clip(clip):
    if not os.path.exists('360p'):
            os.mkdir('360p')
    if not os.path.exists('360p/clips'):
        os.mkdir('360p/clips')
    if not os.path.exists('360p/output'):
        os.mkdir('360p/output')
    
    src = 'clips/' + clip
    tar = '360p/clips/' + clip
    out = '360p/output/' + ''.join(clip.split('.')[:-1]) + '.txt'
    if os.path.exists(tar):
        return
    cmd = 'ffmpeg -i %s -vf scale=640x360 -an -c:v libx265 -tune ssim -tag:v hvc1 \
        -colorspace bt709 -color_primaries bt709 -pix_fmt yuv420p -aspect 16:9 -vsync 0         \
        -x265-params ssim=1:keyint=50:min-keyint=50:vbv-bufsize=2200:vbv-maxrate=1650:scenecut=0:scenecut-bias=0:open-gop=0\
        -crf 26 -y %s 2> %s' % (src, tar, out)
    try:
        os.system(cmd)
    except:
        print('clip error:', end=' ')
        print(tar)
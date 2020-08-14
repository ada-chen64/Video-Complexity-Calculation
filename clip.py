import os


def cut_clip(clip, bitrate=0):
    '''
    360p clips for features and information extraction

    Param:
        clip: absolute path, eg. '/home/vtech/.../clips/clip1.mp4'
    Return:
        base: basename without extent, eg. 'clip1'
    '''

    # get base name
    path, ext = os.path.splitext(clip)
    base = os.path.basename(path)

    
    # ffmpeg bitrate mode
    if bitrate > 0:
        src = clip
        dpi = '720p'
        tar = dpi + '/clips/' + base + '.mp4'
        out = dpi + '/output/' + base + '.txt'
        bufsize = 2 * bitrate
        maxrate = 1.5 * bitrate
        cmd = 'ffmpeg -i %s -vf scale=1280x720 -an -c:v libx265 -tune ssim -tag:v hvc1  \
            -colorspace bt709 -color_primaries bt709 -pix_fmt yuv420p -aspect 16:9 -vsync 0 \
            -x265-params ssim=1:keyint=50:min-keyint=50:vbv-bufsize=%d:vbv-maxrate=%d:scenecut=0:scenecut-bias=0:open-gop=0 \
            -b:v %dk -y %s 2> %s' % (src, bufsize, maxrate, bitrate, tar, out)
    
    # ffmpeg crf mode
    else:
        src = clip
        dpi = '360p'
        tar = dpi + '/clips/' + base + '.mp4'
        out = dpi + '/output/' + base + '.txt'           
        cmd = 'ffmpeg -i %s -vf scale=640x360 -an -c:v libx265 -tune ssim -tag:v hvc1 \
            -colorspace bt709 -color_primaries bt709 -pix_fmt yuv420p -aspect 16:9 -vsync 0 \
            -x265-params ssim=1:keyint=50:min-keyint=50:vbv-bufsize=2200:vbv-maxrate=1650:scenecut=0:scenecut-bias=0:open-gop=0\
            -crf 26 -y %s 2> %s' % (src, tar, out)
    

    # make dictionary
    if not os.path.exists(dpi):
        os.mkdir(dpi)
    if not os.path.exists(dpi + '/clips/'):
        os.mkdir(dpi + '/clips')
    if not os.path.exists(dpi + '/output/'):
        os.mkdir(dpi + '/output/')
    
    
    # cut clips
    if not os.path.exists(tar):
        os.system(cmd)
    
    return base
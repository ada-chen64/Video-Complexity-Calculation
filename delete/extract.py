import os, re, json, sys

datadir = sys.argv[1]

files = os.listdir(datadir)

br = re.compile(" ([0-9\.]*?) kb/s")
ssim = re.compile("SSIM Mean Y: (.*?) ")
stdname = re.compile("clip_[0-9]*")

feat = {}

for file in files:
    name = os.path.join(datadir, file)
    with open(name, "r") as f:
        res = ""
        for x in f:
            if not x.strip() == "":
                res = x
    print(file)
    a = br.search(res)
    b = ssim.search(res)

    rname = stdname.search(file).group(0)
    print(rname, a.group(1), b.group(1))
    bitrate = float(a.group(1))
    ssimv = float(b.group(1))

    feat[rname] = [bitrate, ssimv]

with open("res360.json", "w") as f:
    json.dump(feat, f)
# Readme

视频流媒体一组



## 环境

* python3
* cv2
* numpy
* pandas
* pytorch

注意配置环境的时候有以下两点问题：
* 安装时 pip3 install opencv-python 若安装失败提示缺少skbuild模块，需要先通过 pip3 install cmake
* 安装 cv2 以后，若cv2.VideoCapture()视频读取失败，其实还需要再安装 pip3 install opencv-contrib-python 才能成功读取视频

```shell
安装过慢时
pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## 运行

在根目录外的input.txt中添加视频的绝对路径即可。

```shell
python main.py > output.txt
```



## 统计

比较预测值与标签值的准确率、均方根误差等等

```
python label.py
```



## 目录

### code

* ../input.txt 视频绝对路径
* main.py 整体框架
* clip.py 场景压缩至360p
* feat.py 提取特征，数据归一化
* pred.py 预测复杂度
* train.py 训练模型
* label.py 统计预测值准确率等
* output.txt 绝对路径、复杂度、码率、SSIM、20维特征

### related

* 所有相关资料，包括分割的场景、bitrate-ssim曲线、MSU调研报告、最终答辩展示和报告等
* 详见文件夹中的readme

### 360p

* clips crf模式压缩后视频
* output crf模式压缩信息

### 720p

* clips 固定码率压缩视频
* output 固定码率压缩信息

### data

* norm.npy 提供归一化数据（最大值、最小值）
* train_data.txt 训练集数据：包括第一列label和后面的20维特征
* test_data.txt 测试集数据：包括20维特征（注意这里的测试集是我们组的84个clip）

### model

* net.pkl 保存模型数据

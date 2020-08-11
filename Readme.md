# Readme

视频流媒体一组



## 环境

* python3
* cv2
* numpy
* pandas
* sklearn
* pytorch



## 运行

将需要检测复杂度的场景clip.mp4放入clips文件夹中，并在clips.txt内添加文件名即可。

```shell
python main.py
```



## 目录

### code

* clips.txt 提供场景名
* main.py 整体框架
* clip.py 场景压缩至360p
* feat.py 提取特征，数据归一化
* pred.py 预测复杂度

### data

* norm.npy 提供归一化数据（最大值、最小值）

### model

* net.pkl 保存模型数据

### delete

* 之前大家所提供的各种.py和.csv文件，暂时可以不再需要了
* 其中的pre.py里包括了整合.csv并转为.npy文件的方式



## 补充

对于训练集data矩阵，其中第0列为label，后续为features

对于测试集data矩阵，由于未提供label，因此全部为features

如果测试集能够提供label，可以读取后调用pred.py中的performance函数，获取测试集结果的准确率和均方根误差值，以及表现不佳的测试样例。
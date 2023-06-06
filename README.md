# SSD-paddlepaddle

#第一步：解压缩ssd-model、pretrained-model、mobilenet_v1_imagenet三个压缩包

#测试的话需要在终端运行 python infer.py


#SSD简介
Single Shot MultiBox Detector (SSD) 是一种单阶段的目标检测器。与两阶段的检测方法不同，单阶段目标检测并不进行区域推荐，而是直接从特征图回归出目标的边界框和分类概率。SSD 运用了这种单阶段检测的思想，并且对其进行改进：在不同尺度的特征图上检测对应尺度的目标。如下图所示，SSD 在六个尺度的特征图上进行了不同层级的预测。每个层级由两个3x3卷积分别对目标类别和边界框偏移进行回归。因此对于每个类别，SSD 的六个层级一共会产生 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732 个检测结果。
![Image text](https://github.com/cyy1111-cai/SSD-paddlepaddle/blob/main/ssd.png)


SSD正是利用了来自多个特征图上的信息进行检测的。比如VGG、ResNet、MobileNet这些都属于提取特征的网络。很多时候会叫Backbone。在这个示例中我们使用 MobileNet。
MobileNet-SSD体结构如下图所示:




在训练时还会对图片进行数据增强，包括随机扰动、扩张、翻转和裁剪:

1-扰动: 扰动图片亮度、对比度、饱和度和色相。

2-扩张: 将原始图片放进一张使用像素均值填充(随后会在减均值操作中减掉)的扩张图中，再对此图进行裁剪、缩放和翻转。

3-翻转: 水平翻转。

4-裁剪: 根据缩放比例、长宽比例两个参数生成若干候选框，再依据这些候选框和标注框的面积交并比(IoU)挑选出符合要求的裁剪结果。

参考论文：https://arxiv.org/pdf/1512.02325.pdf



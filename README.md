# SSD-paddlepaddle

#第一步：解压缩ssd-model、pretrained-model、mobilenet_v1_imagenet三个压缩包
#测试的话需要在终端运行 python infer.py


#SSD简介
Single Shot MultiBox Detector (SSD) 是一种单阶段的目标检测器。与两阶段的检测方法不同，单阶段目标检测并不进行区域推荐，而是直接从特征图回归出目标的边界框和分类概率。SSD 运用了这种单阶段检测的思想，并且对其进行改进：在不同尺度的特征图上检测对应尺度的目标。如下图所示，SSD 在六个尺度的特征图上进行了不同层级的预测。每个层级由两个3x3卷积分别对目标类别和边界框偏移进行回归。因此对于每个类别，SSD 的六个层级一共会产生 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732 个检测结果。
[![Uploading image.png…]()](https://github.com/cyy1111-cai/SSD-paddlepaddle/blob/main/ssd.png)

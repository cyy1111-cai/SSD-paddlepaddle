

# -*- coding: UTF-8 -*-
"""
使用训练完成的模型进行预测
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import time
import paddle.fluid as fluid

from PIL import Image
from IPython.display import display
from PIL import ImageDraw

target_size = [3, 224, 224]
nms_threshold = 0.45
confs_threshold = 0.5
place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = "./ssd-model"
[inference_program, feed_target_names, fetch_targets] =     fluid.io.load_inference_model(dirname=path,
                                  params_filename='mobilenet-ssd-params',
                                  model_filename='mobilenet-ssd-model',
                                  executor=exe)
print(fetch_targets)


def draw_bbox_image(img, nms_out, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param nms_out:
    :param save_name:
    :return:
    """
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    for dt in nms_out:
        if dt[1] < confs_threshold:
            continue
        category_id = dt[0]
        bbox = dt[2:]
        xmin, ymin, xmax, ymax = clip_bbox(dt[2:])
        draw.rectangle((xmin * img_width, ymin * img_height, xmax * img_width, ymax * img_height), None, 'red')
    img.save(save_name)
    display(img)


def clip_bbox(bbox):
    """
    截断矩形框
    :param bbox:
    :return:
    """
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    percent_h = float(target_size[1]) / img.size[1]
    percent_w = float(target_size[2]) / img.size[0]
    percent = min(percent_h, percent_w)
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    w_off = (target_size[1] - resized_width) / 2
    h_off = (target_size[2] - resized_height) / 2
    img = img.resize((target_size[1], target_size[2]), Image.ANTIALIAS)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    img = Image.open(img_path)
    resized_img = img.copy()
    img = resize_img(img, target_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return img, resized_img


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    tensor_img, resized_img = read_image(image_path)
    t1 = time.time()
    nmsed_out = exe.run(inference_program,
                        feed={feed_target_names[0]: tensor_img},
                        fetch_list=fetch_targets,
                        return_numpy=False)
    period = time.time() - t1
    print("predict result:{0} cost time:{1}".format(nmsed_out, "%2.2f sec" % period))
    nmsed_out = np.array(nmsed_out[0])
    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-reslut.jpg'
    print("result save to:", out_path)
    draw_bbox_image(resized_img, nmsed_out, out_path)


image_path = 'dog-cat.jpg'
infer(image_path)



import colorsys
import copy
import time
import os  # 用于处理绝对路径

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

# 导入 nets 文件夹下的模型定义
from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class Unet(object):
    # 获取当前文件所在的绝对路径，确保 logs 文件夹能被找到
    _current_path = os.path.dirname(os.path.abspath(__file__))

    _defaults = {
        # 指向实验 01 的权重路径
        "model_path": os.path.join(_current_path, '01_Base_Unet_Exp', 'logs', 'best_epoch_weights.pth'),
        "num_classes": 2,
        "backbone": "vgg",
        "input_shape": [256, 256],
        "mix_type": 0,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 设置分割颜色：背景黑 (0,0,0)，瞳孔红 (128,0,0)
        self.colors = [(0, 0, 0), (128, 0, 0)]

        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"无法找到权重文件: {self.model_path}\n请检查路径是否正确")

        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('成功加载模型权重: {}'.format(self.model_path))

        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        # 此函数用于日常预测和可视化显示
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points = orininal_h * orininal_w
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                classes_nums[i] = num
            if name_classes:
                print(f"检测结果: {name_classes[1]} 像素占比 {classes_nums[1] / total_points * 100:.2f}%")

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))

        return image

    # ---------------------------------------------------#
    #   ⭐ 新增：用于计算 mIoU 的专用预测函数
    # ---------------------------------------------------#
    def get_miou_png(self, image):
        # 将图片转为 RGB
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 预处理：不失真 resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 推理
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 截取灰条
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            # 缩放到原图大小并获取索引
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        # 返回单通道灰度图（0 代表背景，1 代表瞳孔）
        image = Image.fromarray(np.uint8(pr))
        return image
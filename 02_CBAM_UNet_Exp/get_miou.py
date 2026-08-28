import os
import sys
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nets.vgg import VGG16
from nets.attention import CoordAtt
from utils.utils_metrics import compute_mIoU, show_results
from utils.utils import cvtColor, preprocess_input, resize_image


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CoordAtt(out_size, out_size)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.ca(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class AttentionUnetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(AttentionUnetModel, self).__init__()
        self.vgg = VGG16(pretrained=pretrained)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1, feat2, feat3, feat4, feat5 = self.vgg.forward(inputs)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        final = self.final(up1)
        return final


class UnetEvaluator:
    def __init__(self, model_path, num_classes=2, input_shape=[256, 256], cuda=True):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.cuda = cuda

        self.net = AttentionUnetModel(num_classes=num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"无法找到权重文件: {model_path}")

        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net = self.net.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print(f'成功加载模型权重: {model_path}')

    def get_miou_png(self, image):
        image = cvtColor(image)
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

        return Image.fromarray(np.uint8(pr))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    VOCdevkit_path = os.path.join(current_dir, "..", "VOCdevkit")

    model_path = os.path.join(current_dir, "logs", "best_epoch_weights.pth")
    miou_out_path = os.path.join(current_dir, "miou_out")

    num_classes = 2
    name_classes = ["background", "pupil"]

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    print("正在加载模型...")
    evaluator = UnetEvaluator(model_path)
    print("模型加载成功。")

    print("正在生成预测掩膜...")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image = Image.open(image_path)
        result = evaluator.get_miou_png(image)
        result.save(os.path.join(pred_dir, image_id + ".png"))
    print("预测完成。")

    print("正在计算 mIoU...")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
    print("计算完成。")
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

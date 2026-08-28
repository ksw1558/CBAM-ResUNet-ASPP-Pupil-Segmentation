import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------#
#   1. 比例设置
#   train_percent 用于改变 训练集：验证集 的比例 9:1
# -------------------------------------------------------#
trainval_percent = 1.0  # 100% 的数据用于训练和验证
train_percent = 0.9  # 训练集占 90%，验证集占 10%

# -------------------------------------------------------#
#   2. 路径设置
#   确保指向你创建的 VOCdevkit 文件夹
# -------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Step 1: 正在 ImageSets 中生成训练/验证划分文件...")

    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    if not os.path.exists(saveBasePath):
        os.makedirs(saveBasePath)

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list_idx = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_idx, tv)
    train = random.sample(trainval, tr)

    print(f"数据集总量: {num}")
    print(f"训练集数量: {tr}")
    print(f"验证集数量: {tv - tr}")

    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list_idx:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("✅ 划分文件生成完毕。")

    # -------------------------------------------------------#
    #   Step 2: 关键检查 - 确保你的标签值是 0 和 1
    # -------------------------------------------------------#
    print("\nStep 2: 正在检查数据集标签格式（瞳孔分割必须为0和1）...")
    classes_nums = np.zeros([256], dtype=int)

    for i in tqdm(list_idx):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)

        # 使用 Pillow 打开并检查
        png = Image.open(png_file_name)
        png_array = np.array(png)

        # 统计像素值
        classes_nums += np.bincount(png_array.flatten(), minlength=256)

    print("\n当前标签中存在的像素值与对应数量：")
    print('-' * 37)
    print("| %15s | %15s |" % ("像素值(Key)", "像素点数(Value)"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    # --- 核心错误预警 ---
    if classes_nums[255] > 0:
        print("\n❌ 警告：检测到标签像素值为 255！")
        print("瞳孔分割是二分类任务，请确保：")
        print("1. 背景像素值为 0")
        print("2. 瞳孔区域像素值为 1（而不是 255）")
        print("💡 如果你的标签是 0 和 255，请先运行格式转换脚本。")
    elif classes_nums[1] > 0:
        print("\n✅ 检查通过：标签包含像素值 1，符合二分类要求。")

    print("\n所有操作已完成，你可以开始配置训练脚本了。")
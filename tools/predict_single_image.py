import os
import sys
import torch
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# === 1. 智能定位项目根目录 ===
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)

while True:
    if os.path.exists(os.path.join(current_dir, "VOCdevkit")):
        root_path = current_dir
        break
    if os.path.exists(os.path.join(current_dir, "nets")):
        root_path = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        root_path = current_dir
        break
    current_dir = parent

sys.path.append(root_path)
print(f"📂 已定位项目根目录: {root_path}")


def predict_single_image(img_id, model, model_name, output_dir, input_size=256):
    """对单张图片进行预测"""
    # 路径配置
    voc_root = os.path.join(root_path, "VOCdevkit", "VOC2007")
    img_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")
    
    img_path = os.path.join(img_dir, img_id + ".jpg")
    label_path = os.path.join(label_dir, img_id + ".png")
    
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print(f"❌ 图片或标签不存在: {img_id}")
        return False
    
    try:
        # 获取标签原图尺寸
        label_img = Image.open(label_path)
        target_size = label_img.size
        
        # 预处理
        image = Image.open(img_path).convert('RGB')
        image_resized = image.resize((input_size, input_size), Image.BILINEAR)
        
        img_np = np.array(image_resized, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            
            # 检查是否是概率输出
            if output.size(1) == 2:
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            else:
                # 如果是单通道概率图
                pred = (torch.sigmoid(output[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
        
        # Resize 回原图尺寸
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_resized = pred_img.resize(target_size, Image.NEAREST)
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        pred_resized.save(os.path.join(output_dir, img_id + ".png"))
        print(f"✅ {model_name} 预测完成: {img_id}")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 预测失败: {e}")
        return False


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_id = "03_16_1401"
    print(f"🎯 开始预测图片: {img_id}")
    
    # ============ 1. Base U-Net ============
    try:
        from nets.unet import Unet
        model_unet = Unet(num_classes=2)
        model_path_unet = os.path.join(root_path, "01_Base_Unet_Exp", "logs", "best_epoch_weights.pth")
        if os.path.exists(model_path_unet):
            checkpoint = torch.load(model_path_unet, map_location=device)
            if 'net' in checkpoint: checkpoint = checkpoint['net']
            model_unet.load_state_dict(checkpoint, strict=False)
            model_unet.to(device)
            model_unet.eval()
            output_dir_unet = os.path.join(root_path, "01_Base_Unet_Exp", "miou_out", "detection-results")
            predict_single_image(img_id, model_unet, "Base U-Net", output_dir_unet, 256)
    except Exception as e:
        print(f"❌ Base U-Net 加载失败: {e}")
    
    # ============ 2. ResUNet ============
    try:
        from nets.res_unet import ResUnet
        model_resunet = ResUnet(num_classes=2)
        model_path_resunet = os.path.join(root_path, "08_ResUNet_Exp", "logs", "best_epoch_weights.pth")
        if os.path.exists(model_path_resunet):
            checkpoint = torch.load(model_path_resunet, map_location=device)
            if 'net' in checkpoint: checkpoint = checkpoint['net']
            model_resunet.load_state_dict(checkpoint, strict=False)
            model_resunet.to(device)
            model_resunet.eval()
            output_dir_resunet = os.path.join(root_path, "08_ResUNet_Exp", "miou_out", "detection-results")
            predict_single_image(img_id, model_resunet, "ResUNet", output_dir_resunet, 256)
    except Exception as e:
        print(f"❌ ResUNet 加载失败: {e}")
    
    # ============ 3. ACR-UNet (CBAM-ResUNet-ASPP) ============
    try:
        from nets.cbam_res_unet_exp11 import CBAMResUnetExp11
        model_acr = CBAMResUnetExp11(num_classes=2, pretrained=False)
        model_path_acr = os.path.join(root_path, "11_CBAM-ResUNet-ASPP", "logs", "best_epoch_weights.pth")
        if not os.path.exists(model_path_acr):
            # 尝试其他可能的文件名
            alt_paths = [
                os.path.join(root_path, "11_CBAM-ResUNet-ASPP", "logs", "final_exp11_miou98_16_epoch070.pth"),
                os.path.join(root_path, "11_CBAM-ResUNet-ASPP", "logs", "exp12_final.pth")
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    model_path_acr = p
                    break
        
        if os.path.exists(model_path_acr):
            checkpoint = torch.load(model_path_acr, map_location=device)
            if 'net' in checkpoint: checkpoint = checkpoint['net']
            model_acr.load_state_dict(checkpoint, strict=False)
            model_acr.to(device)
            model_acr.eval()
            output_dir_acr = os.path.join(root_path, "11_CBAM-ResUNet-ASPP", "miou_out", "detection-results-optimized-final")
            predict_single_image(img_id, model_acr, "ACR-UNet", output_dir_acr, 256)
    except Exception as e:
        print(f"❌ ACR-UNet 加载失败: {e}")
    
    print("\n🎉 所有预测任务完成！")

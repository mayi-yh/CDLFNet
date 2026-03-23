import torch as t
from RGBT_dataprocessing_CNet import testData1
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from abc_net import MyConv_resnet_T
from model.segnet import SegNet
# from abc_unemsc import MyConv_resnet_T
# from abc_unenhance import MyConv_resnet_T
# from abc_unconvnext import MyConv_resnet_T
# from abc_unswin import MyConv_resnet_T
from model.FCN import FCN8
from model.DeepLab_v3_plus import DeepLabv3_plus
from model.RefineNet.RefineNet import RefineNet
from model.SwinNet import SwinTransformerSys
from model.UCTransNet.UCTransNet import UCTransNet
from model.BEFUnet.BEFUnet import BEFUnet
from model.BEFUnet.configs import get_BEFUnet_configs
from model.EMCAD.networks import EMCADNet
from model.DuAT.DuAT import DuAT
from model.UNeXt.UNeXt import UNext
from model.MTUnet import MTUNet
import torch

# 初始化模型和数据加载器
# configs = get_BEFUnet_configs()
net = MTUNet()
net.load_state_dict(t.load('/root/autodl-tmp/pytorch_foot_ulcer_seg/YPthMTUNet2025_07_08_12_55_last.pth', map_location='cpu'))
test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=1)

# 定义路径（确保这些路径存在）
output_path = '/root/autodl-tmp/pytorch_foot_ulcer_seg/Result/MTUNet/'  # 添加了output_path定义
gt_root = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/test/labels/'

# 创建输出目录（如果不存在）
os.makedirs(output_path, exist_ok=True)

# 评估指标初始化
total_ber = 0.0
total_iou = 0.0
total_mae = 0.0
total_fscore = 0.0
total_precision = 0.0
total_recall = 0.0
total_acc = 0.0
img_count = 0

# 数据转换器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

with t.no_grad():
    net.eval()
    for i, sample in enumerate(test_dataloader1):
        image = sample['RGB']
        name = sample['name'][0]  # 更安全地获取文件名
        # out1, out2, out3 = net(image)
        out3= net(image)
        # out1, out2, out3, out4 = net(image)
        out = t.sigmoid(out3)
        out_img = out.cpu().detach().numpy()[0]  # 去掉batch维度
        
        # 保存预测结果
        save_path = os.path.join(output_path, name + '.png')
        cv2.imwrite(save_path, (out_img[0] * 255).astype(np.uint8))  # 假设单通道输出
        
        # ==================== 指标计算 ====================
        # 1. 加载GT标签
        gt_path = os.path.join(gt_root, name + '.png')
        if not os.path.exists(gt_path):
            print(f"警告: 未找到标签文件 {gt_path}, 跳过此样本")
            continue
            
#         gt = Image.open(gt_path).convert('L')  # 灰度模式
#         gt = transform(gt)  # 转换为tensor [1,224,224]
        
#         # 2. 统一数据格式
#         pred_tensor = t.from_numpy(out_img).float()  # [1,224,224]
#         gt_tensor = gt.float()
        gt_tensor = sample['label'][-1].permute(0, 3, 1, 2)
        pred_tensor = out
        # print('gt_tensor', gt_tensor.shape)
        # print('pred_tensor', pred_tensor.shape)
        # 3. 二值化处理
        pred_bin = (pred_tensor >= 0.5).float()
        gt_bin = (gt_tensor >= 0.5).float()
        
        # 4. 计算指标
        # BER (平衡错误率)
        N_p = t.sum(gt_bin) + 1e-20
        N_n = t.sum(1 - gt_bin) + 1e-20
        TP = t.sum(pred_bin * gt_bin)
        TN = t.sum((1 - pred_bin) * (1 - gt_bin))
        ber = 1 - 0.5 * (TP/N_p + TN/N_n)
        
        # IOU (交并比)
        intersection = t.sum(pred_bin * gt_bin)
        union = t.sum(pred_bin) + t.sum(gt_bin) - intersection
        iou = intersection / (union + 1e-20)
        
        # MAE (平均绝对误差)
        mae = torch.sum(torch.abs(gt_bin - pred_bin)) / (224.0*224.0)
        
        # F-measure, Precision, Recall
        precision = TP / (t.sum(pred_bin) + 1e-20)
        recall = TP / (t.sum(gt_bin) + 1e-20)
        f_score = 2 * precision * recall / (precision + recall + 1e-20)
        
        # Accuracy
        # acc = t.sum(pred_bin == gt_bin).float() / (224*224)
        correct_pixels = torch.sum(pred_bin == gt_bin)
        total_pixels = torch.prod(torch.tensor(gt_bin.shape))
        acc = correct_pixels / total_pixels
        
        # 5. 累加指标
        total_ber += ber.item()
        total_iou += iou.item()
        total_mae += mae.item()
        total_fscore += f_score.item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_acc += acc.item()
        img_count += 1
        
        print(f'[{img_count}/{len(test_dataloader1)}] {name} | '
              f'BER: {ber.item():.4f} | IOU: {iou.item():.4f} | '
              f'F1: {f_score.item():.4f}')

# 6. 计算平均指标
if img_count > 0:
    avg_metrics = {
        'BER': total_ber / img_count,
        'IOU': total_iou / img_count,
        'MAE': total_mae / img_count,
        'F1': total_fscore / img_count,
        'Precision': total_precision / img_count,
        'Recall': total_recall / img_count,
        'Accuracy': total_acc / img_count
    }

    # 7. 打印最终结果
    print('\n============== 最终评估结果 ==============')
    for metric, value in avg_metrics.items():
        print(f'{metric}: {value:.4f}')
else:
    print('警告: 没有处理任何有效样本')
import torch as t
from RGBT_dataprocessing_CNet import testData1  # 你的数据集
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import numpy as np
import cv2

from abc_net import MyConv_resnet_T  # 你的模型定义
# from abc_unemsc import MyConv_resnet_T
from model.DeepLab_v3_plus import DeepLabv3_plus
from model.FCN import FCN8
from model.segnet import SegNet
from model.RefineNet.RefineNet import RefineNet
from model.SwinNet import SwinTransformerSys
from model.UCTransNet.UCTransNet import UCTransNet

# 初始化数据加载器
test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=1)

# 初始化模型
net = MyConv_resnet_T()
net.load_state_dict(t.load('/root/autodl-tmp/pytorch_foot_ulcer_seg/YPthMyConv_resnet_T2025_06_22_18_11_last.pth', map_location='cpu'))

# 保存结果路径
output_path = '/root/autodl-tmp/pytorch_foot_ulcer_seg/Result/lastcopy/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print(f'{output_path} exists')

with t.no_grad():
    net.eval()
    for i, sample in enumerate(test_dataloader1):
        image = sample['RGB']           # (batch, C, H, W)
        label = sample['label']         # 你的标签，没用到可以忽略
        name = "".join(sample['name'])  # 文件名

        image = Variable(image)
        label = Variable(label)

        out1, out2, out3 = net(image)
        # out1= net(image)
        out = t.sigmoid(out3)  # (batch, channels, H, W)

        out_img = out.cpu().detach().numpy()  # numpy格式，形状 (1, C, H, W)

        # 转换成 (H, W, C) 格式给 OpenCV
        out_img = out_img[0]  # 去掉batch维度 (C, H, W)

        # 如果输出是单通道，转换成 (H, W)
        if out_img.shape[0] == 1:
            out_img = out_img[0]  # (H, W)
            # 归一化到0-255
            out_img = (out_img * 255).astype(np.uint8)
        else:
            # 多通道，比如3通道，需要转成 (H, W, C) 并归一化到0-255
            out_img = np.transpose(out_img, (1, 2, 0))  # (H, W, C)
            out_img = (out_img * 255).astype(np.uint8)

        # 保存图片
        save_path = os.path.join(output_path, name + '.png')
        cv2.imwrite(save_path, out_img)
        print(f'Saved {save_path}')
        # print("out_img shape:", out_img.shape) #(224, 224, 3

import torch
import torch.nn as nn
import torch.nn.functional as F
# from VMamba.classification.models.vmamba import vmamba_tiny_s1l8
from torch.nn.parameter import Parameter
import math
from convnext import convnext_tiny
from SwinTransformer import SwinTransformer

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(SeparableConv2d, self).__init__()
   
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=dilation, dilation=dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.relu(self.bn(x))
        return x


class Enhance(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.sigmoid = nn.Sigmoid()
        self.sepconv = SeparableConv2d(hidden_dim, hidden_dim, 3)
        self.sepconv_x3 = SeparableConv2d(hidden_dim, hidden_dim, 3)
        self.sepconv_x4 = SeparableConv2d(hidden_dim, hidden_dim, 3)
        self.relu = nn.ReLU()

    def forward(self, x4, x3):
        x4 = self.conv1x1(x4)  # x4 的尺寸变为 (batch_size, hidden_dim, 2H, 2W)
        
        # 调整 x3 的尺寸，使其与 x4 一致
        if x3.shape[2:] != x4.shape[2:]:
            x3 = F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True)
        
        mul_x3_x4 = x4 * x3
        mul_sig = self.sigmoid(self.sepconv(mul_x3_x4))

        x4 = self.sepconv_x4(mul_sig * x4)
        x3 = self.sepconv_x3(mul_sig * x3)
        out = x4 + x3
        return out
                           

class ScaleAwareBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(ScaleAwareBlock, self).__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        main_feat = self.main_conv(x)
        scale_feat = self.scale_conv(x)
        attn = self.attention(main_feat + scale_feat)
        return main_feat * attn + scale_feat * (1 - attn)

class EMSC(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super(EMSC, self).__init__()
        self.branches = nn.ModuleList([
            ScaleAwareBlock(in_channels, out_channels, rate) for rate in dilation_rates
        ])
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_rates) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, r, d):
        x = (r * d) + r + d  # 保持和ASPP相同的融合方式
        branch_outputs = [branch(x) for branch in self.branches]
        global_feat = self.global_ctx(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat(branch_outputs + [global_feat], dim=1)
        return self.fusion(out)

class MyConv_resnet_T(nn.Module):
    def __init__(self):
        super(MyConv_resnet_T, self).__init__()
        
        # self.mamba = vmamba_tiny_s1l8()
        self.cnn = convnext_tiny()
        self.swin = SwinTransformer(
        embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]
)
        self.swin.load_state_dict(torch.load('/root/autodl-tmp/pytorch_foot_ulcer_seg/swin_base_patch4_window7_224.pth')['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model")
        
        # 修改转置卷积层，加入Dropout
        self.deconv4 = nn.Sequential(nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv3 = nn.Sequential(nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv2 = nn.Sequential(nn.Conv2d(192, 96, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv1 = nn.Sequential(nn.Conv2d(192, 96, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv5 = nn.Sequential(nn.Conv2d(512, 384, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv6 = nn.Sequential(nn.Conv2d(256, 192, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv7 = nn.Sequential(nn.Conv2d(128, 96, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv8 = nn.Sequential(nn.Conv2d(128, 48, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.deconv44 = nn.Sequential(nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # TransBasicConv2d(768, 384),
            # nn.Dropout2d(0.1)  # 加入Dropout
        )
        self.deconv33 = nn.Sequential(nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # TransBasicConv2d(384, 192),
            # nn.Dropout2d(0.1)
        )
        self.deconv22 = nn.Sequential(nn.Conv2d(192, 96, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # TransBasicConv2d(192, 96),
            # nn.Dropout2d(0.1)
        )
        self.deconv11 = nn.Sequential(nn.Conv2d(96, 48, kernel_size=1, stride=1, bias=False),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # TransBasicConv2d(96, 48),
            # nn.Dropout2d(0.1)
        )
        
        # 添加 Enhance 模块
        self.enhance_r3 = Enhance(384, 384)  # 用于 r_add3
        self.enhance_r2 = Enhance(192, 192)  # 用于 r_add2
        self.enhance_r1 = Enhance(96, 96)    # 用于 r_add1
        self.enhance_t3 = Enhance(384, 384)  # 用于 t_add3
        self.enhance_t2 = Enhance(192, 192)  # 用于 t_add2
        self.enhance_t1 = Enhance(96, 96)    # 用于 t_add1
        self.enhance_out = Enhance(96,96)
        
        # 使用EMSC替换原来的ASPP
        self.emsc4 = EMSC(768, 768, [1, 3, 5, 7])
        self.enhance_fusion4 = Enhance(768, 768)
        self.enhance_fusion3 = Enhance(384, 384)
        self.enhance_fusion2 = Enhance(192, 192)
        self.enhance_fusion1 = Enhance(96, 96)
        
        # 在最后的卷积层前加入Dropout
        self.conv1 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(96, 3, kernel_size=3, padding=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(96, 3, kernel_size=3, padding=1, stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(96, 3, kernel_size=3, padding=1, stride=1)
        )
        self.conv4 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, stride=1)
        )
        
        self.conv5 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(1024, 768, kernel_size=3, padding=1, stride=1)
        )
        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(384, 192, kernel_size=3, padding=1, stride=1)
        )
        self.conv7 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, stride=1)
        )
        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 96, kernel_size=3, padding=1, stride=1)
        )
        self.conv9 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 384, kernel_size=3, padding=1, stride=1)
        )
        self.conv10 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 192, kernel_size=3, padding=1, stride=1)
        )
        self.conv11 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 96, kernel_size=3, padding=1, stride=1)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)

    def forward(self, rgb):
        # 前向传播逻辑保持不变
        # r_out1, r_out2, r_out3, r_out4, r_out5 = self.mamba(rgb)
        r_out1, r_out2, r_out3, r_out4, r_out5 = self.swin(rgb)
        t_out1, t_out2, t_out3, t_out4 = self.cnn(rgb)
        
        # print("r_out4 shape:", r_out4.shape)   #1024 7
        # print("r_out3 shape:", r_out3.shape)   #512 14 
        # print("r_out2 shape:", r_out2.shape)   #256 28
        # print("r_out1 shape:", r_out1.shape)   #128 56
        
        # fusion4 = self.enhance_fusion4(self.conv5(r_out4),t_out4) #768 14
        fusion4 = self.emsc4(self.conv5(r_out4), t_out4) # 768 7
        # print("fusion4 shape:", fusion4.shape)
        
        # RGB流程
        r_add3 = self.enhance_r3(self.deconv5(r_out3),self.deconv44(fusion4))  #384 56
        # print("r_add3 shape:", r_add3.shape)
        r_add2 = self.enhance_r2(self.deconv6(r_out2),self.conv6(r_add3)) + self.up1(self.up1(self.deconv3(self.deconv4(fusion4)))) #192 112
        # print("r_add2 shape:", r_add2.shape)
        r_add1 = self.enhance_r1(self.deconv7(r_out1),self.conv7(r_add2)) + self.deconv2(self.up1(self.up1(self.deconv3(self.deconv4(fusion4))))) #96 224
        # print("r_add1 shape:", r_add1.shape)
        
        r_out = self.conv2(r_add1)
        r_out = r_out
        
        
        # RGB 流程
        # r_add3 = self.conv9(r_out3) + self.deconv44(self.conv5(r_out4)) + self.deconv4(t_out4) # (3, 384, 14, 14)
        # print("r_add3 shape:", r_add3.shape)
        # r_add2 = self.conv10(r_out2) + self.deconv3(r_add3) + self.deconv33(self.deconv44(t_out4)) + self.deconv33(self.deconv44(self.conv5(r_out4)))# (3, 192, 28, 28)
        # print("r_add2 shape:", r_add2.shape)
        # r_add1 = self.conv11(r_out1) + self.deconv2(r_add2) + self.deconv22(self.deconv33(self.deconv44(t_out4))) + self.deconv22(self.deconv33(self.deconv44(self.conv5(r_out4))))# (3, 96, 56, 56)
        # print("r_add1 shape:", r_add1.shape)
        
        # r_out = self.conv2(r_add1)   #(3,1,56,56)
        # r_out = self.up1(r_out)   #(3,1,112,112)
        # r_out = self.up1(r_out)
        # r_out = self.up1(r_out)
        
        # Thermal流程
        t_add3 = self.enhance_t3(self.up1(t_out3),self.deconv4(fusion4)) #384 56
        t_add2 = self.enhance_t2(self.up1(t_out2),self.deconv3(t_add3)) + self.up1(self.up1(self.deconv3(self.deconv4(fusion4))))#192 112
        t_add1 = self.enhance_t1(self.up1(t_out1),self.deconv2(t_add2)) + self.deconv22(self.up1(self.up1(self.deconv3(self.deconv4(fusion4)))))#96 224
        
        t_out = self.conv2(t_add1)
        t_out = t_out
        

         # Thermal 流程
#         t_add3 = t_out3 + self.deconv4(t_out4) + self.deconv44(self.conv5(r_out4)) # (3, 384, 14, 14)
#         t_add2 = t_out2 + self.deconv3(t_add3) + self.deconv33(self.deconv44(self.conv5(r_out4)))+ self.deconv33(self.deconv44(t_out4)) # (3, 192, 28, 28)
#         t_add1 = t_out1 + self.deconv2(t_add2) + self.deconv22(self.deconv33(self.deconv44(self.conv5(r_out4)))) + self.deconv22(self.deconv33(self.deconv44(t_out4)))# (3, 96, 56, 56)
        
#         t_out = self.conv2(t_add1)   #(3,1,112,112)
#         t_out = self.up1(t_out)   #(3,1,112,112)
#         t_out = self.up1(t_out)
        # t_out = self.up1(t_out)
        
        # Fusion流程
        # fusion3 = r_add3 + t_add3
        # fusion2 = r_add2 + t_add2
        # fusion2 = fusion2 + self.deconv3(fusion3)
        
        # fusion1 = r_add1 + t_add1
        # fusion1 = fusion1 + self.deconv2(fusion2) + self.deconv2(self.deconv3(fusion3))
        # print("fusion1 shape:", fusion1.shape)
        
        
        
        # out = self.enhance_fusion1(r_add1,t_add1)
        out = self.conv2(r_add1 + t_add1)
        # out = self.up1(out)
        # out = self.up1(out)
        
    
#         fusion3 = self.enhance_fusion3(r_add3, t_add3)  # ([3, 384, 56, 56])
#         # fusion3 = fusion3 + self.up1(fusion4)  # ([3, 384, 112, 112])
        
#         fusion2 = self.enhance_fusion2(r_add2, t_add2)  #([3, 192, 112, 112])
#         fusion2 = fusion2 + self.deconv3(fusion3)  #([3, 192, 224, 224])
        
#         fusion1 = self.enhance_fusion1(r_add1, t_add1)  # ([3, 96, 224, 224])
#         fusion1 = fusion1 + self.deconv2(self.deconv3(fusion3))  #([3, 96, 448, 448])
        
#         # 最终输出
        # out = self.conv2(fusion1)   #(3,1,56,56)
        # out = self.up1(out)
        # out = self.up1(out)
#         out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)  # 上采样到 224x224
        
        # return t_out, r_out, out
        return t_out, r_out, out


if __name__ == '__main__':
    # Ensure the device is set to GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create random RGB and thermal input tensors with shape (batch_size, channels, height, width)
    rgb_input = torch.randn(3, 3, 224, 224).to(device)  # Move to correct device
    # thermal_input = torch.randn(3, 3, 224, 224).to(device)  # Move to correct device
    
    # Create model instance and move to the same device as the inputs
    model = MyConv_resnet_T().to(device)  # Use .to() to move model to the correct device
    
    # Call the forward method of the model
    t_out, r_out, out = model(rgb_input)
    
    # Print output shapes
    # print("r_out1 shape:", r_out1.shape)
    # print("r_out2 shape:", r_out2.shape)
    # print("r_out3 shape:", r_out3.shape)
    # print("r_out4 shape:", r_out4.shape)
    # print("r_out5 shape:", r_out5.shape)
    # print("t_out1 shape:", t_out1.shape)
    # print("t_out2 shape:", t_out2.shape)
    # print("t_out3 shape:", t_out3.shape)
    # print("t_out4 shape:", t_out4.shape)
    # print("t_out5 shape:", t_out5.shape)
    print("t_out shape:", t_out.shape)
    print("r_out shape:", r_out.shape)
    print("out shape:", out.shape)
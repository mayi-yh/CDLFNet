import torch
from torch import nn
# from heatmap_generator import HeatmapGenerator
from RGBT_dataprocessing_CNet import trainData, valData
# from RGBT_dataprocessing_CNet_SOD import trainData, valData
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
# import Loss.lovasz_losses as lovasz
import pytorch_iou
import sys
sys.path.append('/root/autodl-tmp/pytorch_foot_ulcer_seg/')
from abc_net import MyConv_resnet_T
# from abc_unemsc import MyConv_resnet_T
from abc_unconvnext import MyConv_resnet_T
from model.DeepLab_v3_plus import DeepLabv3_plus
from model.FCN import FCN8
from model.segnet import SegNet
from model.RefineNet.RefineNet import RefineNet
from model.SwinNet import SwinTransformerSys
# from model.TransUNet.TransUNet import VisionTransformer
from model.RefineNet.RefineNet import RefineNet
from model.UCTransNet.UCTransNet import UCTransNet
from model.EMCAD.networks import EMCADNet
# from model.resnet_my import paper3
import time
import os
from log import get_logger
# from  Loss.Binary_Dice_loss import BinaryDiceLoss
# from Loss.Focal_loss import sigmoid_focal_loss
# import pytorch_ssim
import numpy as np

def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params +=p.numel()
    print(name)
    print("The number of parameters:{}M".format(num_params/1000000))


IOU = pytorch_iou.IOU(size_average = True).cuda()

class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

################################################################################################################
batchsize = 8
################################################################################################################

train_dataloader = DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)

test_dataloader = DataLoader(valData,batch_size=batchsize,shuffle=True,num_workers=4)


# net = paper3('vgg16').cuda()
config = {
    'img_size': 224,
    'patch_size': 16,
    'in_channels': 3,
    'embed_dim': 768,
    'num_heads': 12,
    # ... 其他参数 ...
}
net = MyConv_resnet_T().cuda()
# net.load_pre('/home/hjk/文档/test_models_FPS/backbone/Shunted/ckpt_B.pth')
# # net.load_state_dict(torch.load('/home/hjk/文档/third_model_GCN/Pth/_GORCNet_SOD_path_RGBT_320_2022_10_31_13_48_best.pth'))
# net = net.cuda()

################################################################################################################
model = 'MyConv_resnet_T' + time.strftime("%Y_%m_%d_%H_%M")
print_network(net, model)
################################################################################################################
bestpath = '/root/autodl-tmp/pytorch_foot_ulcer_seg/YPth' + model + '_best.pth'
lastpath = '/root/autodl-tmp/pytorch_foot_ulcer_seg/YPth' + model + '_last.pth'
################################################################################################################
criterion1 = BCELOSS().cuda()
criterion2 = BCELOSS().cuda()
criterion3 = BCELOSS().cuda()
criterion4 = BCELOSS().cuda()
criterion5 = BCELOSS().cuda()
criterion6 = BCELOSS().cuda()
criterion7 = BCELOSS().cuda()
criterion8 = BCELOSS().cuda()

# focaloss = sigmoid_focal_loss().cuda()
# diceloss = BinaryDiceLoss().cuda()

criterion_val = BCELOSS().cuda()
################################################################################################################
lr_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=lr_rate, weight_decay=1e-3)
################################################################################################################

best = [10]
step = 0
mae_sum = 0
best_mae = 1
best_epoch = 0

logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)

logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')

################################################################################################################
epochs = 150
################################################################################################################

logger.info(f'Epochs:{epochs}  Batchsize:{batchsize}')
def accuary(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size
for epoch in range(epochs):
    mae_sum = 0
    trainmae = 0
    if (epoch+1) % 20 == 0 and epoch != 0:
        for group in optimizer.param_groups:
            group['lr'] = 0.5 * group['lr']
            print(group['lr'])
            lr_rate = group['lr']


    train_loss = 0
    net = net.train()
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        # image = Variable(sample['RGB'].float().cuda())
        thermal = Variable(sample['thermal'].cuda())
        label = Variable(sample['label'].float().cuda())
        # label2 = Variable(sample['label2'].float().cuda())
        # label3 = Variable(sample['label3'].float().cuda())
        # label4 = Variable(sample['label4'].float().cuda())
        # label5 = Variable(sample['label5'].float().cuda())
        # bound = Variable(sample['bound'].float().cuda())
        optimizer.zero_grad()
        # out1 = net(image, depth)
        # out1, out2, out3, out4, s1, s2, s3, s4 = net(image, depth)
        # out1, out2, out3, e1, e2, e3 = net(image, depth)
        # print(image.shape)
        # print(thermal.shape)
        t_out, r_out, out = net(image)
        # out1, out2, out3, out4 = net(image)
        # out = net(image)

        # print(out1.shape)
        # print(label.shape)

        t_out = F.sigmoid(t_out)
        r_out = F.sigmoid(r_out)
        out = F.sigmoid(out)
        # out1 = F.sigmoid(out1)
        # out2 = F.sigmoid(out2)
        # out3 = F.sigmoid(out3)
        # out4 = F.sigmoid(out4)
        # out4 = F.sigmoid(out4)
        # out5 = F.sigmoid(out5)
        # label /= (label.max() + 1e-8)
        # print(label.shape)
        # e1 = F.sigmoid(e1)
        # e2 = F.sigmoid(e2)
        # e3 = F.sigmoid(e3)

        loss1 = criterion1(r_out, label) + IOU(r_out, label)
        loss2 = criterion1(out, label) + IOU(out, label)
        loss3 = criterion1(t_out, label) + IOU(t_out, label)
        # loss1 = criterion1(out1, label) + IOU(out1, label)
        # loss2 = criterion1(out2, label) + IOU(out2, label)
        # loss3 = criterion1(out3, label) + IOU(out3, label)
        # loss4 = criterion1(out4, label) + IOU(out4, label)
        # loss4 = criterion1(out4, label4) + IOU(out4, label4)
        # loss5 = criterion1(out5, label5) + IOU(out5, label5)
        # loss1 = criterion1(out1, label)


        # eloss1 = criterion1(e1, bound) + IOU(e1, bound)
        # eloss2 = criterion1(e2, bound) + IOU(e2, bound)
        # eloss3 = criterion1(e3, bound) + IOU(e3, bound)


        loss_total =  loss1 + loss2 + loss3
        # loss_total =  loss1 + loss2 + loss3 + loss4
        # loss_total =loss2
        # loss_total = loss + iou_loss


        time = datetime.now()

        if i % 10 == 0 :
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch, epochs, i, len(train_dataloader), loss_total.item(), loss2))
        loss_total.backward()
        optimizer.step()
        train_loss = loss_total.item() + train_loss


    net = net.eval()
    eval_loss = 0
    mae = 0

    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):

            imageVal = Variable(sampleTest['RGB'].cuda())
            thermalVal = Variable(sampleTest['thermal'].cuda())
            labelVal = Variable(sampleTest['label'].float().cuda())

            # bound = Variable(sampleTest['bound'].float().cuda())
            
            
            # input_folder = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/train/images'
            # output_folder = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/train/heatmaps'

            # 创建 HeatmapGenerator 类的实例
            # heatmap_generator = HeatmapGenerator(input_folder, output_folder)

            # 调用 process_all_images 方法，处理文件夹中的所有图像并生成热力图
            # Thermal = heatmap_generator.process_all_images()

            # out1 = net(imageVal, depthVal)
            
            t_out, r_out, out1 = net(imageVal)
            # out1, out2, out3, out4 = net(imageVal)
            # out1 = net(imageVal)
            
            # out1, out2, out3, out4, edge_all, edge1, edge2, edge3, edge4 = net(imageVal, depthVal)
            # out = F.sigmoid(out1[13])
            out = F.sigmoid(out1)
            # out = out1[0]
            loss = criterion_val(out, labelVal)

            # outVal = outVal[0].max(dim=1)[1].data.cpu().numpy()
            maeval = torch.sum(torch.abs(labelVal - out)) / (224.0*224.0)

            print('===============', j, '===============', loss.item())
    #
    #         # if j==34:
    #         #     out=out[4].cpu().numpy()
    #         #     edge = edge[4].cpu().numpy()
    #         #     out = out.squeeze()567
    #         #     edge = edge.squeeze()
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/out.png', out,cmap='gray')
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/edge1.png', edge,cmap='gray')
    #
            eval_loss = loss.item() + eval_loss
            mae = mae + maeval.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = '{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    logger.info(
        f'Epoch:{epoch+1:3d}/{epochs:3d} || trainloss:{train_loss / 1500:.8f} valloss:{eval_loss / 362:.8f} || '
        f'valmae:{mae / 362:.8f} || lr_rate:{lr_rate} || spend_time:{time_str}')

    if (mae / 362) <= min(best):
        best.append(mae / 362)
        nummae = epoch+1
        torch.save(net.state_dict(), bestpath)

    torch.save(net.state_dict(), lastpath)
    print('=======best mae epoch:{},best mae:{}'.format(nummae, min(best)))
    logger.info(f'best mae epoch:{nummae:3d}  || best mae:{min(best)}')















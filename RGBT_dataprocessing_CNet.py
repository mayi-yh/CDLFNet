import torch as t
import os
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import matplotlib
import numpy as np
import random
import torchvision





# gt 为png  其他两个为jpg格式
# Train NJU2K+LFSD
path_rgbt_RGB = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/train/images'
lr = os.listdir(path_rgbt_RGB)
lr = [os.path.join(path_rgbt_RGB, img) for img in lr]
lr.sort()
# print(lr)

path_rgbt_GT = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/train/labels'
gt = os.listdir(path_rgbt_GT)
gt = [os.path.join(path_rgbt_GT, gtimg) for gtimg in gt]
gt.sort()

path_rgbt_T = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/train/heatmaps'
thermal = os.listdir(path_rgbt_T)
thermal = [os.path.join(path_rgbt_T, dep) for dep in thermal]
thermal.sort()
#
# path_rgbt_bound = '/media/hjk/shuju/轨道缺陷检测/Dataset/train/boundary'
# bound = os.listdir(path_rgbt_bound)
# bound = [os.path.join(path_rgbt_bound, edge) for edge in bound]
# bound.sort()




# val NJU2K+LFSD
path_rgbt_val_rgb = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/validation/images'
lrval = os.listdir(path_rgbt_val_rgb)
lrval = [os.path.join(path_rgbt_val_rgb, img) for img in lrval]
lrval.sort()

path_rgbt_val_GT = '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/validation/labels'
gtval = os.listdir(path_rgbt_val_GT)
gtval = [os.path.join(path_rgbt_val_GT, gtimg) for gtimg in gtval]
gtval.sort()

path_rgbt_val_T= '/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/validation/heatmaps'
thermalval = os.listdir(path_rgbt_val_T)
thermalval = [os.path.join(path_rgbt_val_T, dep) for dep in thermalval]
thermalval.sort()



###Test
VT800_RGB = os.listdir('/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/test/images')
VT800_RGB = [os.path.join('/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/test/images', img) for img in VT800_RGB]
VT800_RGB.sort()

VT800_GT = os.listdir('/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/test/labels')
print(len(VT800_GT))
VT800_GT = [os.path.join('/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/test/labels', gtimg) for gtimg in VT800_GT]
VT800_GT.sort()

VT800_T = os.listdir('/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/test/heatmaps')
VT800_T = [os.path.join('/root/autodl-tmp/pytorch_foot_ulcer_seg/dataset/test/heatmaps', dep) for dep in VT800_T]
VT800_T.sort()

############################################

# VT1000_RGB = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/RGB')
# VT1000_RGB = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/RGB', img) for img in VT1000_RGB]
# VT1000_RGB.sort()


# VT1000_GT = os.listdir('/home/hjk/文档/轨道缺陷检测/GT')
# VT1000_GT = [os.path.join('/home/hjk/文档/轨道缺陷检测/GT', gtimg) for gtimg in VT1000_GT]
# VT1000_GT.sort()


# VT1000_T = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/T')
# VT1000_T = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/T', dep) for dep in VT1000_T]
# VT1000_T.sort()



############################################
# VT5000_RGB = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/RGB')
# VT5000_RGB = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/RGB', img) for img in VT5000_RGB]
# VT5000_RGB.sort()
#
#
# VT5000_GT = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/GT')
# VT5000_GT = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/GT', gtimg) for gtimg in VT5000_GT]
# VT5000_GT.sort()
#
#
# VT5000_T = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/T')
# VT5000_T = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/T', dep) for dep in VT5000_T]
# VT5000_T.sort()



class NJUDateset(Dataset):
    def __init__(self, train, transform=None):
        self.train = train
        if self.train:
            self.lrimgs = lr
            self.thermal = thermal
            self.gt = gt
            # self.bound = bound


        else:
            self.lrimgs = lrval
            self.thermal = thermalval
            self.gt = gtval
            # self.bound = bound

        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.lrimgs[index]
        thermalPath = self.thermal[index]
        gtPath = self.gt[index]
        # gt_b = self.bound[index]
        img = Image.open(imgPath)  # 0到255
        img = np.asarray(img)
        thermal = Image.open(thermalPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
        thermal = np.asarray(thermal)
        gt = Image.open(gtPath) # 0,255
        gt = np.asarray(gt).astype(np.float)
        if gt.max() == 255.:
            gt = gt / 255.
        # gt_b = Image.open(gt_b)  # 0,255
        # gt_b = np.asarray(gt_b).astype(np.float)
        # if gt_b.max() == 255.:
        #     bound = gt_b / 255.

        # sample = {'RGB': img, 'label': gt}
        sample = {'RGB': img, 'thermal': thermal, 'label': gt}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.lrimgs)

#
class TEST1(Dataset):
    def __init__(self, test, transform=None):
        self.test = test
        if self.test:
            self.lrimgs = VT800_RGB
            # self.thermalPath = VT800_T
            self.gt = VT800_GT


        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.lrimgs[index]
        # # thermalPath = self.thermal[index]
        gtPath = self.gt[index]
        img = Image.open(imgPath)  # 0到255
        img = np.asarray(img)
        # thermal = Image.open(thermalPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
        # thermal = np.asarray(thermal)
        gt = Image.open(gtPath)  # 0,255
        gt = np.asarray(gt).astype(np.float)
        # print(gt.shape)
        if gt.max() == 255.:
            gt = gt / 255.
        name = imgPath.split('/')[-1].split('.')[-2]
        sample = {'RGB': img,'label': gt, 'name': name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.lrimgs)



#
# class TEST2(Dataset):
#     def __init__(self, test, transform=None):
#         self.test = test
#         if self.test:
#
#
#             self.lrimgs = VT1000_RGB
#             self.depth = VT1000_T
#             self.gt = VT1000_GT
#             # #
#
#
#         self.transform = transform
#
#     def __getitem__(self, index):
#         imgPath = self.lrimgs[index]
#         # imgboundpath = self.imgbound[index]
#         depthPath = self.depth[index]
#         gtPath = self.gt[index]
#         img = Image.open(imgPath)  # 0到255
#         img = np.asarray(img)
#         depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
#         depth = np.asarray(depth)
#         gt = Image.open(gtPath)  # 0,255
#         gt = np.asarray(gt).astype(np.float)
#         # print(gt.shape)
#         if gt.max() == 255.:
#             gt = gt / 255.
#         name = imgPath.split('/')[-1].split('.')[-2]
#         sample = {'RGB': img, 'depth': depth, 'label': gt, 'name': name}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
#
#     def __len__(self):
#         return len(self.lrimgs)

# class TEST3(Dataset):
#     def __init__(self, test, transform=None):
#         self.test = test
#         if self.test:
#
#             self.lrimgs = VT5000_RGB
#             self.depth = VT5000_T
#             self.gt = VT5000_GT
#
#
#
#         self.transform = transform
#
#     def __getitem__(self, index):
#         imgPath = self.lrimgs[index]
#         # imgboundpath = self.imgbound[index]
#         depthPath = self.depth[index]
#         gtPath = self.gt[index]
#         img = Image.open(imgPath)  # 0到255
#         img = np.asarray(img)
#         depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
#         depth = np.asarray(depth)
#         gt = Image.open(gtPath)  # 0,255
#         gt = np.asarray(gt).astype(np.float)
#         # print(gt.shape)
#         if gt.max() == 255.:
#             gt = gt / 255.
#         name = imgPath.split('/')[-1].split('.')[-2]
#         sample = {'RGB': img, 'depth': depth, 'label': gt, 'name': name}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
#
#     def __len__(self):
#         return len(self.lrimgs)

########################################################
image_h = 224
image_w = 224
# image_h = 256
# image_w = 256

class scaleNorm(object):
    def __call__(self, sample):
        # image, label = sample['RGB'], sample['label']
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']

        # Bi-linear
        image = cv2.resize(image, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        thermal = cv2.resize(thermal, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        label = cv2.resize(label, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        # bound = cv2.resize(bound, (image_h, image_w), interpolation=cv2.INTER_NEAREST)
        #
        #
        # return {'RGB': image,  'label': label}
        return {'RGB': image, 'thermal': thermal, 'label': label}


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['RGB']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)


        # return {'RGB': img_new,  'label': sample['label']}
        return {'RGB': img_new, 'thermal': sample['thermal'], 'label': sample['label']}


class RandomFlip(object):
    def __call__(self, sample):
        # image,  label= sample['RGB'], sample['label']
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            thermal = np.fliplr(thermal).copy()
            label = np.fliplr(label).copy()
            # bound = np.fliplr(bound).copy()
        #
        # return {'RGB': image, 'label': label}
        return {'RGB': image, 'thermal': thermal, 'label': label}


# Transforms on torch.*Tensor

class RandomRotate(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Rotate image
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Rotate thermal
        thermal = cv2.warpAffine(thermal, M, (w, h))
        
        # Rotate label
        label = cv2.warpAffine(label, M, (w, h))
        
        return {'RGB': image, 'thermal': thermal, 'label': label}
    
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]
        thermal = thermal[top:top + new_h, left:left + new_w]
        label = label[top:top + new_h, left:left + new_w]

        return {'RGB': image, 'thermal': thermal, 'label': label}
    
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample):
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']
        
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
        
        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=saturation_factor, beta=0)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return {'RGB': image, 'thermal': thermal, 'label': label}
    
class GaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']
        noise = np.random.normal(self.mean, self.std, image.shape) * 255
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return {'RGB': noisy_image, 'thermal': thermal, 'label': label}
    
class RandomScale(object):
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, sample):
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        thermal = cv2.resize(thermal, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return {'RGB': image, 'thermal': thermal, 'label': label}


class Normalize(object):
    def __call__(self, sample):
        image,thermal, label = sample['RGB'], sample['thermal'], sample['label']
        image = image / 255.0
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        # if depth.max() > 256.0:
        #     depth = depth / 31197.0
        # else:
        thermal = thermal / 255.0
        thermal = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(thermal)
        label = label

        sample['RGB'] = image
        sample['thermal'] = thermal
        sample['label'] = label

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, label = sample['RGB'], sample['label']
        image, thermal, label = sample['RGB'], sample['thermal'], sample['label']

        # Generate different label scales
        label2 = cv2.resize(label, (image_h // 2, image_w // 2), interpolation=cv2.INTER_NEAREST)

        label3 = cv2.resize(label, (image_h // 4, image_w // 4), interpolation=cv2.INTER_NEAREST)

        label4 = cv2.resize(label, (image_h // 8, image_w // 8), interpolation=cv2.INTER_NEAREST)

        label5 = cv2.resize(label, (image_h // 16, image_w // 16), interpolation=cv2.INTER_NEAREST)

        label6 = cv2.resize(label, (image_h // 32, image_w // 32), interpolation=cv2.INTER_NEAREST)
        # bound1 = cv2.resize(bound, (image_h // 2, image_w // 2), interpolation=cv2.INTER_NEAREST)
        #
        # bound2 = cv2.resize(bound, (image_h // 4, image_w // 4), interpolation=cv2.INTER_NEAREST)
        #
        # bound3 = cv2.resize(bound, (image_h // 8, image_w // 8), interpolation=cv2.INTER_NEAREST)
        #
        # bound4 = cv2.resize(bound, (image_h // 16, image_w // 16), interpolation=cv2.INTER_NEAREST)
        #
        # bound5 = cv2.resize(bound, (image_h // 32, image_w // 32), interpolation=cv2.INTER_NEAREST)
        #


        # swap color axis because
        # numpy RGB: H x W x C
        # torch RGB: C X H X W
        image = image.transpose((2, 0, 1))

        thermal = thermal.transpose((2, 0 ,1))

        # depth.squeeze()
        # depth = np.array([depth,depth,depth])
        # depth = np.array([depth])
        # depth = depth / 1.0
        # print(depth.shape)
        # depth = depth.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        label2 = label2.transpose((2, 0, 1))
        label3 = label3.transpose((2, 0, 1))
        label4 = label4.transpose((2, 0, 1))
        label5 = label5.transpose((2, 0, 1))
        label6 = label6.transpose((2, 0, 1))
        # label = label
        # print('label', label.shape)

        # label = np.expand_dims(label, 0).astype(np.float)
        # print(label.shape)
        # bound = np.expand_dims(bound, 0).astype(np.float)
        # bound1 = np.expand_dims(bound1, 0).astype(np.float)
        # bound2 = np.expand_dims(bound2, 0).astype(np.float)
        # bound3 = np.expand_dims(bound3, 0).astype(np.float)
        # bound4 = np.expand_dims(bound4, 0).astype(np.float)
        # bound5 = np.expand_dims(bound5, 0).astype(np.float)
        # label2 = np.expand_dims(label2, 0).astype(np.float)
        # label3 = np.expand_dims(label3, 0).astype(np.float)
        # label4 = np.expand_dims(label4, 0).astype(np.float)
        # label5 = np.expand_dims(label5, 0).astype(np.float)
        # label6 = np.expand_dims(label6, 0).astype(np.float)
        return {'RGB': t.from_numpy(image).float(),
                'thermal': t.from_numpy(thermal).float(),
                'label': t.from_numpy(label).float(),
                'label2': t.from_numpy(label2).float(),
                'label3': t.from_numpy(label3).float(),
                'label4': t.from_numpy(label4).float(),
                'label5': t.from_numpy(label5).float(),
                'label6': t.from_numpy(label6).float(),
                # 'bound': t.from_numpy(bound).float(),
                # 'bound1': t.from_numpy(bound1).float(),
                # 'bound2': t.from_numpy(bound2).float(),
                # 'bound3': t.from_numpy(bound3).float(),
                # 'bound4': t.from_numpy(bound4).float(),
                # 'bound5': t.from_numpy(bound5).float(),
                }

class scaleNormtest(object):
    def __call__(self, sample):
        image, label, name = sample['RGB'], sample['label'], sample['name']

        # Bi-linear
        image = cv2.resize(image, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        # thermal = cv2.resize(thermal, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        label = cv2.resize(label, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        name = name

        return {'RGB': image, 'label': label, 'name': name}



class Normalizetest(object):
    def __call__(self, sample):
        image, label, name = sample['RGB'], sample['label'], sample['name']
        image = image / 255.0
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        # if depth.max() > 256.0:
        #     depth = depth / 31197.0
        # else:
        # thermal = thermal / 255.0
        # thermal = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(thermal)

        label = label
        name = name
        sample['RGB'] = image
        sample['label'] = label
        sample['name'] = name
        return sample



class ToTensortest(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image,  label, name = sample['RGB'], sample['label'], sample['name']

        # Generate different label scales
        label2 = cv2.resize(label, (image_h // 2, image_w // 2), interpolation=cv2.INTER_NEAREST)

        label3 = cv2.resize(label, (image_h // 4, image_w // 4), interpolation=cv2.INTER_NEAREST)

        label4 = cv2.resize(label, (image_h // 8, image_w // 8), interpolation=cv2.INTER_NEAREST)

        label5 = cv2.resize(label, (image_h // 16, image_w // 16), interpolation=cv2.INTER_NEAREST)

        label6 = cv2.resize(label, (image_h // 32, image_w // 32), interpolation=cv2.INTER_NEAREST)



        # swap color axis because
        # numpy RGB: H x W x C
        # torch RGB: C X H X W
        image = image.transpose((2, 0, 1))
        # thermal = thermal.transpose((2, 0, 1))
        # depth = np.array([depth])
        # depth = depth / 1.0
        label = np.expand_dims(label, 0).astype(np.float)
        # print(label.shape)
        # label2 = np.expand_dims(label2, 0).astype(np.float)
        # label3 = np.expand_dims(label3, 0).astype(np.float)
        # label4 = np.expand_dims(label4, 0).astype(np.float)
        # label5 = np.expand_dims(label5, 0).astype(np.float)
        # label6 = np.expand_dims(label6, 0).astype(np.float)
        return {'RGB': t.from_numpy(image).float(),
                # 'thermal': t.from_numpy(thermal).float(),
                'label': t.from_numpy(label).float(),
                # 'label2': t.from_numpy(label2).float(),
                # 'label3': t.from_numpy(label3).float(),
                # 'label4': t.from_numpy(label4).float(),
                # 'label5': t.from_numpy(label5).float(),
                # 'label6': t.from_numpy(label6).float(),
                'name': name
                }


trainData = NJUDateset(train=True, transform=torchvision.transforms.Compose([
    scaleNorm(),
    RandomHSV((0.9, 1.1),
              (0.9, 1.1),
              (25, 25)),
    RandomFlip(),
    ToTensor(),
    Normalize()
]))

valData = NJUDateset(train=False, transform=torchvision.transforms.Compose([
    scaleNorm(),
    ToTensor(),
    Normalize(),
]
))


testData1 = TEST1(test=True, transform=torchvision.transforms.Compose([
    scaleNormtest(),
    ToTensortest(),
    Normalizetest(),
]
))

# testData2 = TEST2(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData3 = TEST3(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))

# testData4 = TEST4(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData5 = TEST5(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData6 = TEST6(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData7 = TEST7(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision
    sample = trainData[100]
    # name = sample['name']
    l1 = sample['label']
    # l2 = sample['label2']
    # l3 = sample['label3']
    # l4 = sample['label4']
    # l5 = sample['label5']
    # l6 = sample['label6']
    img = sample['RGB']
    # depth = sample['depth']
    img1 = torchvision.transforms.ToPILImage()(l1)

    plt.imshow(img1)
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL转cv2
    # plt.imshow('RGB', out_img)
    # cv2.imshow('rgb',img)
    # bound = sample['bound']
    # bound2 = sample['bound2']
    # bound3 = sample['bound3']
    # bound4 = sample['bound4']
    # bound5 = sample['bound5']
    # print(np.max(depth))
    # print(name)
    # print(l2.shape)
    # print(img.shape)
    # print(depth.shape)
    # print(depth)
    # print(bound.shape)
    # import numpy as np
    # uni1 = np.unique(l1)
    # print(uni1)
    # uni1 = np.unique(l2)
    # print(uni1)
    # uni1 = np.unique(l3)
    # print(uni1)
    # uni1 = np.unique(l4)
    # print(uni1)
    # uni1 = np.unique(l5)
    # print(uni1)
    # uni1 = np.unique(l6)
    # print(uni1)
    # bound = np.unique(bound)
    # print(bound)
    # bound2 = np.unique(bound2)
    # print(bound2)
    # bound3 = np.unique(bound3)
    # print(bound3)
    # bound4 = np.unique(bound4)
    # print(bound4)
    # bound5 = np.unique(bound5)
    # print(bound5)

# #


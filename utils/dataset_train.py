import copy

import cv2
import numpy as np
from torch.utils.data import Dataset


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def ImageNew(src):
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    result = usm
    return result


def Image_GaussianBlur(img):
    kernel_size = (5, 5)
    sigma = 1.5
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    return img


class dataset_train(Dataset):
    def __init__(self, csv, num_classes, input_shape, isRandom=True, label="label"):
        super(Dataset, self).__init__()
        self.csv = csv
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.isRandom = isRandom
        self.label = label

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, item):
        pic_train = cv2.imread(self.csv.loc[item, "path"])
        if self.isRandom:
            pic_train = self.get_random_data(pic_train, self.input_shape)
        pic_train = np.transpose(cv2.cvtColor(pic_train, cv2.COLOR_BGR2RGB), [2, 0, 1])
        pic_label = np.array(self.csv.loc[item, self.label])
        return pic_train / 255.0, pic_label

    @staticmethod
    def resize_cv2(image, input_size):
        ih, iw = input_size
        h, w = image.shape[:2]
        image_mask = np.ones([ih, iw, 3], dtype=image.dtype) * 128
        if iw / ih < w / h:
            nw = copy.copy(iw)
            nh = int(h / w * nw)
            mask = 1
        else:
            nh = ih
            nw = int(w / h * nh)
            mask = 0
        if (image == 0).all():
            image = cv2.resize(image, (nw, nh))
        else:
            image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
        if mask == 1:
            image_mask[int((ih - nh) / 2):int((ih - nh) / 2) + nh, :, :] = image
        else:
            image_mask[:, int((iw - nw) / 2):int((iw - nw) / 2) + nw, :] = image
        return image_mask

    def get_random_data(self, image, input_shape, random=True):
        image = cvtColor(image)
        h, w = image.shape[0], image.shape[1]
        ih, iw = input_shape
        if random:
            #   生成随机数，scale负责随机缩放、锐化、高斯模糊，scale flip 负责上下左右旋转
            scale = rand(0, 1)
            scale_flip = rand(0, 1)
            #   随机缩放、锐化、高斯模糊
            if scale < 0.25:
                new_ar = iw / ih * rand(1, 2) / rand(1, 2)
                nh = h
                nw = int(h * new_ar)
                if (image == 0).all():
                    image = cv2.resize(image, (nw, nh))
                else:
                    image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
            elif 0.25 <= scale < 0.5:
                image = Image_GaussianBlur(image)
            elif 0.5 <= scale < 0.75:
                image = ImageNew(image)

            #   随机旋转
            if scale_flip < 0.25:
                image = cv2.flip(image, -1)
            elif 0.25 <= scale_flip < 0.5:
                image = cv2.flip(image, 0)
            elif 0.5 <= scale_flip < 0.75:
                image = cv2.flip(image, 1)

        #   将图像多余的部分加上灰条
        image = self.resize_cv2(image, input_shape)
        return image

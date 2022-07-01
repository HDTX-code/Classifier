import math

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ClassDataset(Dataset):
    def __init__(self, annotation_lines, train=True, transforms=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.train = train
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()
        image_path = line[0]
        label = int(float(line[1]))
        img = self.Pre_pic(image_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def Pre_pic(self, pic_path):
        png = cv2.imread(pic_path)
        if not (png == 0).all():
            png = png * 5
            png[png > 255] = 255
            png = self.gamma_trans(png, math.log10(0.5) / math.log10(np.mean(png[png > 0]) / 255))
        image = Image.fromarray(cv2.cvtColor(png, cv2.COLOR_BGR2RGB)).convert('RGB')
        return image

    @staticmethod
    def gamma_trans(img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    def get_height_and_width(self, index):
        line = self.annotation_lines[index].split()
        h, w = int(line[2]), int(line[3])
        return h, w

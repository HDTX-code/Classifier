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
        label = int(line[1])
        img = Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def get_height_and_width(self, index):
        line = self.annotation_lines[index].split()
        h, w = int(line[2]), int(line[3])
        return h, w

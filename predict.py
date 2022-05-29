import copy
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_model, dataset_predict


def go_pre(args):
    # 训练设备
    print("GPU: ", end="")
    print(torch.cuda.is_available())
    print("")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 特征网络
    print("backbone = " + args.backbone)

    # 检查保存文件夹是否存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 生成提交csv
    class_df = pd.DataFrame(columns=["id", "path", "class_predict"])

    # 加载模型
    model = get_model(args.backbone, args.model_path, args.num_classes)

    # 加载dataloader
    data_list = []
    if os.path.exists(os.path.join(args.pic_path, 'test')):
        path_root = os.path.join(args.pic_path, 'test')
        for item_case in os.listdir(path_root):
            for item_day in os.listdir(os.path.join(path_root, item_case)):
                path = os.path.join(path_root, item_case, item_day, 'scans')
                data_list.extend(map(lambda x: os.path.join(path, x), os.listdir(path)))
    else:
        path_root = os.path.join(args.pic_path, 'train')
        for item_case in os.listdir(path_root):
            for item_day in os.listdir(os.path.join(path_root, item_case)):
                path = os.path.join(path_root, item_case, item_day, 'scans')
                data_list.extend(map(lambda x: os.path.join(path, x), os.listdir(path)))
            break

    class_df["path"] = data_list
    class_df["id"] = class_df["path"].apply(lambda x: str(x.split("/")[5]) + "_" + str(
        x.split("/")[-1].split("_")[0] + '_' + x.split("/")[-1].split("_")[1]))
    dataset = dataset_predict(copy.copy(class_df), [args.h, args.w])
    gen = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # 开始预测
    with tqdm(total=len(gen), mininterval=0.3) as pbar:
        with torch.no_grad():
            model.eval().to(device)
            for item, (pic, label_item) in enumerate(gen):
                png = png.type(torch.FloatTensor).to(device)
                output = model(png)
                for item_batch in range(output.shape[0]):
                    pr = output[item_batch].argmax(axis=-1).cpu().numpy()
                    class_df.loc[label_item, "class_predict"] = pr[0]
                pbar.update(1)

    # 保存结果
    class_df.to_csv(os.path.join(args.save_dir, "class_predict.csv"))
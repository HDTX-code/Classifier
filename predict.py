import argparse
import copy
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets import get_score
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
    class_df = pd.read_csv(args.class_df_path)
    class_df_dataset = copy.deepcopy(class_df)

    # 加载模型
    model = get_model(args.backbone, args.model_path, args.num_classes)

    # dataloader
    dataset = dataset_predict(class_df_dataset, [args.h, args.w], args.is_pre, label=args.label)
    gen = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # 开始预测
    if args.label is None:
        with tqdm(total=len(gen), mininterval=0.3) as pbar:
            with torch.no_grad():
                model.eval().to(device)
                for item, (pic, label_item) in enumerate(gen):
                    pic = pic.type(torch.FloatTensor).to(device)
                    output = model(pic)
                    label_item = label_item.cpu().numpy()
                    for item_batch in range(output.shape[0]):
                        pr = output[item_batch].argmax().cpu().numpy()
                        class_df.loc[label_item[item_batch], "class_predict"] = pr
                    pbar.update(1)
    else:
        with tqdm(total=len(gen), mininterval=0.3, postfix=dict) as pbar:
            with torch.no_grad():
                model.eval().to(device)
                Score = 0
                for item, (pic, label_item, pic_label) in enumerate(gen):
                    pic = pic.type(torch.FloatTensor).to(device)
                    pic_label = pic_label.long().to(device)
                    output = model(pic)
                    score = get_score(output, pic_label)
                    Score += score.item()
                    pbar.set_postfix(**{'s': Score / (item + 1)})
                    label_item = label_item.cpu().numpy()
                    for item_batch in range(output.shape[0]):
                        pr = output[item_batch].argmax().cpu().numpy()
                        class_df.loc[label_item[item_batch, 0], "class_predict"] = pr
                    pbar.update(1)

    # 保存结果
    class_df.to_csv(os.path.join(args.save_dir, "class_predict.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预测设置')
    parser.add_argument('--backbone', type=str, default='resnet18', help='特征网络选择，默认resnet18')
    parser.add_argument('--num_classes', type=int, default=3, help='种类数量')
    parser.add_argument('--save_dir', type=str, default="./", help='存储文件夹位置')
    parser.add_argument('--model_path', type=str,
                        default=r"D:\work\project\Kaggle_uw\data\weights\class_weights\ep012-score0.953-val_score0.957.pth",
                        help='模型参数位置')
    parser.add_argument('--class_df_path', type=str,
                        default=r"D:\work\project\Kaggle_uw\data\weights\class_weights\csv\val_csv.csv", help='预测csv路径')
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--is_pre', type=bool, default=False, help="是否预处理")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--label', type=str, default='class', help="标签列名")
    parser.add_argument('--w', type=int, default=256, help='宽')
    parser.add_argument('--h', type=int, default=256, help='高')
    args = parser.parse_args()

    go_pre(args)

    # data = pd.read_csv("./class_predict.csv")
    # data_raw = pd.read_csv(r"D:\work\project\Kaggle_uw\data\weights\class_weights\csv\val_csv.csv")
    # print(data["class"].value_counts())
    # print(data["class_predict"].value_counts())
    # print(data.loc[data["class_predict"] == data["class"], "class_predict"].value_counts())
    # print(sum(data["class_predict"] == data_raw["class"]))

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from nets import get_lr_scheduler, set_optimizer_lr
from utils import get_model, dataset_train, LossHistory, fit_one_epoch


def go_train_classes(args):
    # 训练设备
    print("GPU: ", end="")
    print(torch.cuda.is_available())
    print("")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + args.backbone)
    print("Init_lr = " + str(args.Init_lr))
    if args.cls_weights is None:
        cls_weights = np.ones([args.num_classes], np.float32)
    else:
        cls_weights = np.array(args.cls_weights, np.float32)
    print('cls_weights = ', end='')
    print(cls_weights)
    print('')

    # 检查保存文件夹是否存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载模型
    model = get_model(args.backbone, args.model_path, device, args.num_classes)

    # 生成loss_history
    loss_history = LossHistory(args.save_dir, model, input_shape=[args.h, args.w])

    # 生成dataset
    train_csv = pd.read_csv(args.train_csv_path)
    train_dataset = dataset_train(train_csv, args.num_classes, [args.h, args.w], label=args.label)
    if args.val_csv_path is not None:
        val_csv = pd.read_csv(args.val_csv_path)
        val_dataset = dataset_train(val_csv, args.num_classes, [args.h, args.w], isRandom=False, label=args.label)
    else:
        val_dataset = None

    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), args.Init_lr, betas=(args.momentum, 0.999),
                           weight_decay=args.weight_decay),
        'sgd': optim.SGD(model.parameters(), args.Init_lr, momentum=args.momentum, nesterov=True,
                         weight_decay=args.weight_decay)
    }[args.optimizer_type]

    print("-----------------start UnFreeze Train-----------------")
    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, args.Init_lr, args.Init_lr * 0.01, args.epoch)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                     num_workers=args.num_workers)
    if args.val_csv_path is not None:
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size,
                             num_workers=args.num_workers)
    else:
        gen_val = None
    for epoch_now in range(args.epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch_now)
        fit_one_epoch(model=model,
                      optimizer=optimizer,
                      epoch_now=epoch_now,
                      epoch_Freeze=0,
                      epoch_all=args.epoch,
                      gen=gen,
                      gen_val=gen_val,
                      save_dir=args.save_dir,
                      cls_weights=cls_weights,
                      device=device,
                      loss_history=loss_history,
                      num_classes=args.num_classes,
                      interval=args.save_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--num_classes', type=int, default=3, help='种类数量')
    parser.add_argument('--save_dir', type=str, default="./logs", help='存储文件夹位置')
    parser.add_argument('--save_interval', type=int, default=3, help='存储间隔')
    parser.add_argument('--model_path', type=str, default="", help='模型参数位置')
    parser.add_argument('--w', type=int, default=512, help='宽')
    parser.add_argument('--h', type=int, default=512, help='高')
    parser.add_argument('--train_csv_path', type=str, default="./data_csv.csv", help="训练csv")
    parser.add_argument('--val_csv_path', type=str, required=True, help="验证csv")
    parser.add_argument('--optimizer_type', type=str, default='adam', help="优化器")
    parser.add_argument('--batch_size', type=int, default=8, help="训练batch_size")
    parser.add_argument('--lr_decay_type', type=str, default='cos', help="使用到的学习率下降方式，可选的有'step','cos'")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--Init_lr', type=float, default=1e-4, help="最大学习率")
    parser.add_argument('--momentum', type=float, default=0.9, help="优化器动量")
    parser.add_argument('--weight_decay', type=float, default=0, help="权值衰减，使用adam时建议为0")
    parser.add_argument('--epoch', type=int, default=6, help="解冻训练轮次")
    parser.add_argument('--cls_weights', nargs='+', type=float, default=None, help='交叉熵loss系数')
    parser.add_argument('--label', type=str, default='label', help="标签列名")
    args = parser.parse_args()

    go_train_classes(args)

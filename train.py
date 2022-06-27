import argparse
import os
import datetime
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import ClassDataset
from utils.plot_curve import plot_loss_and_lr, plot_acc
from utils.train_one_epoch import train_one_epoch, evaluate
from utils.utils import get_transform, get_dataloader_with_aspect_ratio_group, get_model, get_lr_fun, set_optimizer_lr


def main(args):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                       训练相关准备                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
    # 检查保存文件夹是否存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 用来保存训练以及验证过程中信息
    results_file = os.path.join(log_dir, "results.txt")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                    训练参数设置相关准备                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    torch.cuda.set_device(args.GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_workers
    args.num_workers = min(min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8]),
                           args.num_workers)
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 权重
    if args.cls_weights is None:
        args.cls_weights = np.ones([args.num_classes], np.float32)
    else:
        assert len(args.cls_weights) == int(args.num_classes), 'len num_classes must eq len cls_weights'
        args.cls_weights = np.array(args.cls_weights, np.float32)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                 dataset dataloader model                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    with open(args.train, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # dataset
    train_dataset = ClassDataset(train_lines, train=True, transforms=get_transform(train=True, mean=mean,
                                                                                   std=std, crop_size=args.size))
    val_dataset = ClassDataset(val_lines, train=False, transforms=get_transform(train=False, mean=mean,
                                                                                std=std, crop_size=args.size))
    # 是否按图片相似高宽比采样图片组成batch, 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor != -1:
        gen = get_dataloader_with_aspect_ratio_group(train_dataset, args.aspect_ratio_group_factor,
                                                     args.batch_size, args.num_workers)
    else:
        gen = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=args.num_workers)
    gen_val = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=args.num_workers)

    # model初始化
    model = get_model(num_classes=args.num_classes, backbone=args.backbone, pretrained=args.pretrained)
    model.to(device)

    # 获取lr下降函数
    lr_scheduler_func, Init_lr_fit, Min_lr_fit = get_lr_fun(args.optimizer_type,
                                                            args.batch_size,
                                                            args.Init_lr,
                                                            args.Min_lr,
                                                            args.Epoch,
                                                            args.lr_decay_type,
                                                            Auto=True)

    # 记录loss lr acc
    train_loss = []
    learning_rate = []
    val_acc = []

    best_acc = 0.
    start_time = time.time()

    print(args)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = {
        'adam': optim.Adam(params, Init_lr_fit, betas=(args.momentum, 0.999), weight_decay=0),
        'sgd': optim.SGD(params, Init_lr_fit, momentum=args.momentum,
                         nesterov=True, weight_decay=args.weight_decay)
    }[args.optimizer_type]

    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.Init_Epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_Epoch = args.Init_Epoch + 1 if args.resume != '' else 1

    print("---------start Train---------")
    for epoch in range(start_Epoch, args.Epoch + 1):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        mean_loss, lr = train_one_epoch(model, optimizer, gen, device, epoch, max_epoch=args.Epoch + 1,
                                        cls_weights=args.cls_weights, scaler=scaler, CE=args.CE,
                                        print_freq=int((num_train / args.batch_size) // 5))
        acc = evaluate(model, gen_val, device=device, print_freq=int(num_val // 5))
        train_loss.append(mean_loss)
        learning_rate.append(lr)
        val_acc.append(acc)
        print('loss: {:.3f}'.format(mean_loss) + '\n' + 'acc: {:.3f}'.format(acc))
        # write into txt
        with open(results_file, "a") as f:
            f.write('loss_{:.3f}'.format(mean_loss) + '\n' + 'acc_{:.3f}'.format(acc) + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            if best_acc < val_acc[-1]:
                torch.save(save_file, os.path.join(log_dir,
                                                   "best_model_{}.pth".format(args.backbone)))
                best_acc = val_acc[-1]
                print('save best acc {}'.format(val_acc[-1]))
        else:
            torch.save(save_file, os.path.join(log_dir,
                                               "{}_epoch_{}_acc_{}.pth".format(args.backbone, epoch, acc)))

    print("---------End UnFreeze Train---------")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, log_dir)
    if len(val_acc) != 0:
        plot_acc(val_acc, log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameter setting')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--save_dir', type=str, default="./weights")
    parser.add_argument('--resume', type=str, default="", help='resume')
    parser.add_argument('--GPU', type=int, default=0, help='GPU_ID')
    parser.add_argument('--size', type=int, default=384, help='pic size')
    parser.add_argument('--train', type=str, default=r"weights/val.txt", help="train_txt_path")
    parser.add_argument('--val', type=str, default=r"weights/val.txt", help="val_txt_path")
    parser.add_argument('--optimizer_type', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--aspect_ratio_group_factor', type=int, default=3)
    parser.add_argument('--lr_decay_type', type=str, default='cos', help="'step' or 'cos'")
    parser.add_argument('--num_workers', type=int, default=24, help="num_workers")
    parser.add_argument('--Init_lr', type=float, default=1e-4, help="max lr")
    parser.add_argument('--Min_lr', type=float, default=1e-6, help="min lr")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0, help="adam is 0")
    parser.add_argument('--Epoch', type=int, default=36)
    parser.add_argument('--Init_Epoch', type=int, default=0, help="Init_Epoch")
    parser.add_argument('--pretrained', default=False, action='store_true', help="pretrained")
    parser.add_argument('--amp', default=True, action='store_true', help="amp or Not")
    parser.add_argument('--save_best', default=True, action='store_true', help="save best or save all")
    parser.add_argument('--CE', default=False, action='store_true', help="CE loss or Focal loss")
    parser.add_argument('--cls_weights', nargs='+', type=float, default=None, help='交叉熵loss系数')
    args = parser.parse_args()

    main(args)





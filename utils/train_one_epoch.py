import torch
from torch import nn
import utils.distributed_utils as utils


def CE_Loss(inputs, target, loss_weight=None, ignore_index: int = -100):
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    return loss


def Focal_Loss(inputs, target, loss_weight, ignore_index: int = -100, alpha=0.5, gamma=2):
    logpt = -nn.CrossEntropyLoss(weight=loss_weight, ignore_index=ignore_index, reduction='none')(inputs,
                                                                                                  target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, cls_weights,
                    print_freq=10, scaler=None, CE=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    with torch.no_grad():
        cls_weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)

    for images, labels in metric_logger.log_every(data_loader, print_freq, header):
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            image, labels = image.to(device), labels.to(device).long()
            output = model(images)
            if CE:
                loss = CE_Loss(output, labels, loss_weight=cls_weights)
            else:
                loss = Focal_Loss(output, labels, loss_weight=cls_weights)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    acc = 0
    with torch.no_grad():
        for images, labels in metric_logger.log_every(data_loader, 100, header):
            image, labels = image.to(device), labels.to(device)
            output = model(images)
            predict_y = torch.max(output, dim=1)[1]
            acc += torch.eq(predict_y, labels.to(device)).sum().item()
    return acc





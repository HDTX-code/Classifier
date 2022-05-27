import numpy as np
import torch

from nets import BasicBlock, ResNet18


def load_model(model, model_path, device):
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    a = {}
    no_load = 0
    for k, v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v):
                a[k] = v
            else:
                no_load += 1
        except:
            pass
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print("No_load: {}".format(no_load))
    print('Finished!')
    return model


def get_model(backbone, model_path, device, num_classes):
    if backbone == 'resnet18':
        model = ResNet18(BasicBlock, num_classes=num_classes).to(device)
    else:
        model = ResNet18(BasicBlock, num_classes=num_classes).to(device)
    if model_path != "":
        model = load_model(model, model_path, device)
    return model


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

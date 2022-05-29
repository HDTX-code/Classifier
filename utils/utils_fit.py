import os

import torch
from tqdm import tqdm

from nets import Focal_Loss, CE_Loss, get_score
from utils import get_lr


def fit_one_epoch(model, optimizer, epoch_now, epoch_Freeze, num_classes,
                  epoch_all, gen, gen_val, save_dir, cls_weights, device,
                  loss_history, interval, focal_loss=True):
    print('Start Train')
    with tqdm(total=len(gen), desc=f'Epoch {epoch_now + 1}/{epoch_all}', postfix=dict, mininterval=0.3) as pbar_train:
        total_loss = 0
        total_score = 0
        model.train().to(device)
        for iteration, (pic_train, label) in enumerate(gen):
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)
                pic_train = pic_train.type(torch.FloatTensor).to(device)
                label = label.long().to(device)

            optimizer.zero_grad()
            outputs = model(pic_train)
            loss = 0

            loss += CE_Loss(outputs, label, weights, num_classes=num_classes)

            if focal_loss:
                loss += Focal_Loss(outputs, label, weights, num_classes=num_classes)

            with torch.no_grad():
                # -------------------------------#
                #   计算score
                # -------------------------------#
                score = get_score(outputs, label)

            # loss.backward()
            # optimizer.step()

            total_loss += loss.item()
            total_score += score.item()
            pbar_train.set_postfix(**{'l': total_loss / (iteration + 1),
                                      's': total_score / (iteration + 1),
                                      'r': get_lr(optimizer)})
            pbar_train.update(1)

    print('Finish Train')
    if gen_val is not None:
        print('Start Validation')
        with tqdm(total=len(gen_val), desc=f'Epoch {epoch_now + 1}/{epoch_all}', postfix=dict,
                  mininterval=0.3) as pbar_val:
            val_loss = 0
            val_score = 0
            model.eval().to(device)
            with torch.no_grad():
                for iteration, (pic_train, label) in enumerate(gen_val):
                    weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)
                    pic_train = pic_train.type(torch.FloatTensor).to(device)
                    label = label.long().to(device)

                    outputs = model(pic_train)

                    loss = 0

                    loss += CE_Loss(outputs, label, weights, num_classes=num_classes)

                    if focal_loss:
                        loss += Focal_Loss(outputs, label, weights, num_classes=num_classes)

                    # -------------------------------#
                    #   计算score
                    # -------------------------------#
                    score = get_score(outputs, label)

                    val_loss += loss.item()
                    val_score += score.item()

                    pbar_val.set_postfix(**{'l': val_loss / (iteration + 1),
                                            's': val_score / (iteration + 1),
                                            'r': get_lr(optimizer)})
                    pbar_val.update(1)
        # 保存模型
        with torch.no_grad():
            print('Finish Validation')
            loss_history.append_loss(epoch_now + 1, total_loss / len(gen), val_loss / len(gen_val))
            print('Epoch:' + str(epoch_now + 1) + '/' + str(epoch_all))
            print('Total Loss: %.6f || Val Loss: %.6f ' % (total_loss / len(gen), val_loss / len(gen_val)))
            print('Total score: %.6f || Val score: %.6f ' % (
                total_score / len(gen), val_score / len(gen_val)))
            if ((epoch_now + 1) % interval == 0 or epoch_now + 1 == epoch_all) and epoch_now >= epoch_Freeze:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-score%.3f-val_score%.3f.pth' % (
                    (epoch_now + 1), total_score / len(gen), val_score / len(gen_val))))
    else:
        with torch.no_grad():
            print('Finish Validation')
            loss_history.append_loss(epoch_now + 1, total_loss / len(gen), 0)
            print('Epoch:' + str(epoch_now + 1) + '/' + str(epoch_all))
            print('Total Loss: %.6f' % (total_loss / len(gen)))
            print('Total score: %.6f ' % (total_score / len(gen)))
            if ((epoch_now + 1) % interval == 0 or epoch_now + 1 == epoch_all) and epoch_now >= epoch_Freeze:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-score%.3f.pth' % (
                    (epoch_now + 1), total_score / len(gen))))
    return model

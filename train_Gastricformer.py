import argparse
import os
import time
from dataset.Dataset import Tumor_Dataset
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from train_utils import *
from swin_models.swin_transformer import SwinTransformer
#from swin_models.swinconv2 import SwinConv2 as ourmodel

from swin_models.swinconv import SwinConv
import losses
from utils import str2bool

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

torch.cuda.set_device(0)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="Test",
                        help='project_name')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default=' ')
    parser.add_argument('--dim', metavar='dim', default=48, type=int)
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='number of classes')
    parser.add_argument('--image_size', default=224, type=int,
                        help='image size')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='pretrained model')

    # loss
    parser.add_argument('--loss', default='CE', choices=LOSS_NAMES)

    # dataset
    parser.add_argument('--img_dir', default="F:\swintransformer\Data",
                        help='dataset name')
    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=5e-05, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-05, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)
    config = parser.parse_args()

    return config


# args = parser.parse_args()

def main():
    config = vars(parse_args())
    # create model
    # model = SwinConv2(
    #     hidden_dim=config['dim'],
    #     layers=(2, 2, 6, 2),
    #     heads=(3, 6, 12, 24),
    #     channels=3,
    #     num_classes=config['num_classes'],
    #     head_dim=32,
    #     window_size=7,
    #     downscaling_factors=(4, 2, 2, 2),
    #     relative_pos_embedding=True
    # )
    model = SwinTransformer(hidden_dim=config['dim'], layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), num_classes=config['num_classes'])
    time_str = time.strftime("%m-%d-%H-%M")
    # config['name'] = "Test"
    config['name'] = f"{type(model).__name__}_{config['dim']}_{time_str}"
    save_dir = f'outputs/{config["name"]}'
    os.makedirs(save_dir, exist_ok=True)
    print('-' * 20)
    with open(f'outputs/{config["name"]}/parameters.txt', 'w', encoding='utf-8') as f:
        for key in config:
            print('%s: %s' % (key, config[key]))
            f.write('%s: %s\n' % (key, config[key]))
    print('-' * 20)

    if config['loss'] == 'CE':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    train_dataset = Tumor_Dataset(split='train', transform=data_transform['train'],dataset_path=config['img_dir'])
    val_dataset = Tumor_Dataset(split='val', transform=data_transform['val'],dataset_path=config['img_dir'])
    test_dataset = Tumor_Dataset(split='test', transform=data_transform['test'], dataset_path=config['img_dir'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=config['num_workers'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers'])
    val_best_ACC_1 = 0
    test_best_ACC_1 = 0
    best_epoch_id = 0
    writer = SummaryWriter(f"./runs/{config['name']}")
    for epoch in range(1, config['epochs'] + 1):
        train_scores = train(config, train_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('train/Loss', train_scores["Loss"], epoch)
        writer.add_scalar('train/ACC@1', train_scores["ACC@1"], epoch)
        # evaluate on validation set
        val_scores = validate(config, val_loader, model, criterion, epoch)
        writer.add_scalar('val/Loss', val_scores["Loss"], epoch)
        writer.add_scalar('val/ACC@1', val_scores["ACC@1"], epoch)
        # Test on Test set
        test_scores = test(config, test_loader, model, criterion, epoch)
        writer.add_scalar('test/Loss', test_scores["Loss"], epoch)
        writer.add_scalar('test/ACC@1', test_scores["ACC@1"], epoch)

        print('train_score:ACC@1:{:.4f},val_score:ACC@1:{:.4f}'.format(train_scores['ACC@1'],val_scores['ACC@1']))
        print('train_score:Loss:{:.4f},val_score:Loss:{:.4f}'.format(train_scores['Loss'],val_scores['Loss']))

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        writer.add_scalar('train/train_lr', lr, epoch)
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        if val_scores["ACC@1"] > val_best_ACC_1:
            val_best_ACC_1 = val_scores["ACC@1"]
            best_epoch_id = epoch
            with open(f'outputs/{config["name"]}/val_scores.txt', 'w', encoding='utf-8') as f:
                for key in val_scores:
                    f.write('%s: %s\n' % (key, val_scores[key]))
            save_model(epoch, val_best_ACC_1, best_epoch_id, model, optimizer, scheduler, save_dir, "best_model.pth")
        if test_scores["ACC@1"] > test_best_ACC_1:
            test_best_ACC_1 = test_scores["ACC@1"]
            with open(f'outputs/{config["name"]}/test_scores.txt', 'w', encoding='utf-8') as f:
                for key in val_scores:
                    f.write('%s: %s\n' % (key, test_scores[key]))
            save_model(epoch, test_best_ACC_1, epoch, model, optimizer, scheduler, save_dir, "Test_best_model.pth")
        save_model(epoch, val_best_ACC_1, best_epoch_id, model, optimizer, scheduler, save_dir, "last_model.pth")
        print(f"********** the best val ACC@1 at peoch{best_epoch_id}, "
              f"best val ACC@1:{round(val_best_ACC_1, 4)}***********")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

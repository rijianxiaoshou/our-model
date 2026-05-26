import os
import torch
import random
from tqdm import tqdm
from torch import nn
from torchvision import transforms
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def train(config, train_loader, model, criterion, optimizer, epoch):
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    pbar = tqdm(train_loader, desc=f"Train epoch:[{epoch}/{config['epochs']}]")
    for i_batch, sampled_batch in enumerate(pbar):
        images, labels, path = sampled_batch
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), labels.size(0))
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        acc1_meter.update(acc1.item(), labels.size(0))

    train_scores = {'Loss': loss_meter.avg, 'ACC@1': acc1_meter.avg}
    pbar.close()
    return train_scores


def validate(config, val_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val epoch:[{epoch}/{config['epochs']}]")
        for i_batch, sampled_batch in enumerate(pbar):
            images, labels, path = sampled_batch
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), labels.size(0))
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_meter.update(acc1.item(), labels.size(0))
        val_scores = {'Loss': loss_meter.avg, 'ACC@1': acc1_meter.avg}
        pbar.close()

    return val_scores


def test(config, test_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Test epoch:[{epoch}/{config['epochs']}]")
        for i_batch, sampled_batch in enumerate(pbar):
            images, labels, path = sampled_batch
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), labels.size(0))
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_meter.update(acc1.item(), labels.size(0))
        test_scores = {'Loss': loss_meter.avg, 'ACC@1': acc1_meter.avg}
        pbar.close()

    return test_scores


def save_model(epoch, best_f1_1, best_epoch_id, model, optimizer, scheduler, save_dir, model_name):
    torch.save({
        'epoch_id': epoch,
        'best_val_F1': best_f1_1,
        'best_epoch_id': best_epoch_id,
        'model_G_state_dict': model.state_dict(),
        'optimizer_G_state_dict': optimizer.state_dict(),
        'exp_lr_scheduler_G_state_dict': scheduler.state_dict(),
    }, os.path.join(save_dir, model_name))
    print(f"************ saved {model_name}*************")


def load_pretrained(pretrained_path, model):
    print(f"==============> Loading weight {pretrained_path} for fine-tuning......")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            print("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            # torch.nn.init.constant_(model.head.bias, 0.)
            # torch.nn.init.constant_(model.head.weight, 0.)
            # torch.nn.init.normal_(model.head.bias)
            # torch.nn.init.normal_(model.head.weight)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    print(f"=> loaded successfully '{pretrained_path}'")

    del checkpoint
    torch.cuda.empty_cache()


def LoadPretrain_SimCLR(pretrained_path, model):
    dim_mlp = model.hidden_dim * 8
    model.mlp_head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.mlp_head)
    checkpoint = torch.load(pretrained_path, map_location='cpu')['state_dict']
    model.load_state_dict({k.replace('backbone.', ''): v for k, v in checkpoint.items()})
    model.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_mlp),
            nn.Linear(dim_mlp, 3))
    print(f"=> loaded successfully '{pretrained_path}'")
    del checkpoint
    torch.cuda.empty_cache()
    return model
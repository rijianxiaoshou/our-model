# -*- coding: utf-8 -*-
import sys
import torch
from torchvision import transforms
from tqdm import tqdm
import os
from dataset.Dataset import Tumor_Dataset
from swin_models.swin_transformer import SwinTransformer
# from swin_models.swin_transformer_v2 import SwinTransformerV2
batch_size = 32

weight_path = r"F:\swintransformer\Finnal\outputs\SwinTransformer_48_12-15-10-08\best_model.pth"
# torch.cuda.set_device(0)


def main(dataset_path='F:\swintransformer\Data'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    save_path = f"./outputs"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_transform = {
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_dataset = Tumor_Dataset(split='test', rand=False, transform=data_transform["test"], dataset_path=dataset_path)
    test_num = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("using {} images for testing".format(test_num))
    net = SwinTransformer(
        hidden_dim=48,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=3,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    )
    # net = SwinTransformerV2(num_classes=3)
    net.load_state_dict(
        torch.load(weight_path, map_location='cpu')[
            'model_G_state_dict'])
    net.to(device)
    print(f"model:{type(net).__name__}, dataset:{dataset_path}")
    test_num = len(test_dataset)
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels, path = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    test_accurate = acc / test_num
    print(f'{type(net).__name__} Finished Testing, Best_acc: {test_accurate}')


if __name__ == '__main__':
    main()

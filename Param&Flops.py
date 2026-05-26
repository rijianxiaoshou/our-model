from thop import profile
import torch
from swin_models.swinconv import SwinConv

img = torch.randn(1, 3, 224, 224)
print("-" * 20, "SwinTransformer", "-" * 20)
model = SwinConv(
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=3,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    )
macs, params = profile(model, inputs=(img,), verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# print("-" * 20, "SwinTransformer", "-" * 20)
# model = SwinTransformerV2(num_classes=3)
# macs, params = profile(model, inputs=(img,), verbose=False)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

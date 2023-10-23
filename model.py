import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=10):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def create_model(num_classes=2, min_size=640, max_size=640):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights='DEFAULT',
        trainable_backbone_layers=1
    )
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32),

    )
    model.head.classification_head.loss = FocalLoss()
    model.transform.min_size = min_size
    model.transform.max_size = max_size
    return model


if __name__ == '__main__':
    model = create_model()
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class ResNet50_2C(torch.nn.Module):
    def __init__(self):
        super(ResNet50_2C, self).__init__()
        weights = FCN_ResNet50_Weights.DEFAULT
        self.resnet50 = torchvision.models.segmentation.fcn_resnet50(
            weights=weights, pretrained=True, num_classes=21)
        # Replace the heads to have 2 output channels
        self.resnet50.classifier = torchvision.models.segmentation.fcn.FCNHead(2048, 2)
        self.resnet50.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(1024, 2)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, 256, 256)
        output_2channels_tsr = self.resnet50(input_tsr)['out']  # (N, 2, 256, 256)
        output_tsr = torch.nn.functional.softmax(output_2channels_tsr, 1)[:, 1, :, :].unsqueeze(1)  # (N, 1, 256, 256)
        return output_tsr
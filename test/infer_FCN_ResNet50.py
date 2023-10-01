import logging
import argparse
import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import cv2
import numpy as np
import architectures.resnet50_2c
import sys
import os
from PIL import Image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    imageFilepath,
    neuralNetworkFilepath,
    outputDirectory
):
    logging.info("infer_FCN_ResNet50.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    process_size = 520

    # Load the neural network
    neural_net = architectures.resnet50_2c.ResNet50_2C()
    neural_net.load_state_dict(torch.load(neuralNetworkFilepath))
    neural_net.to(device)
    neural_net.eval()

    # Load the transformation pipeline
    weights = FCN_ResNet50_Weights.DEFAULT
    transforms = weights.transforms(resize_size=process_size)

    # Load the image
    pil_img = Image.open(imageFilepath)
    original_img_size = pil_img.size
    #pil_img = pil_img.resize(process_size)
    img_tsr = transforms(pil_img).float().to(device)

    # Pass the tensor through the neural network
    output_tsr = neural_net(img_tsr.unsqueeze(0))
    output_heatmap = 255 * output_tsr.squeeze(0).squeeze(0).cpu().detach().numpy()
    # Resize to the original size
    output_heatmap = cv2.resize(output_heatmap, original_img_size, interpolation=cv2.INTER_NEAREST)
    logging.debug(f"output_heatmap.shape = {output_heatmap.shape}")
    heatmap_filepath = os.path.join(outputDirectory, "heatmap.png")
    cv2.imwrite(heatmap_filepath, output_heatmap)
    logging.info(f"Wrote the generated heatmap to '{heatmap_filepath}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imageFilepath', help="The image filepath")
    parser.add_argument('neuralNetworkFilepath', help="The neural network filepath")
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_infer_FCN_ResNet50'", default='./output_infer_FCN_ResNet50')
    args = parser.parse_args()

    main(
        args.imageFilepath,
        args.neuralNetworkFilepath,
        args.outputDirectory
    )
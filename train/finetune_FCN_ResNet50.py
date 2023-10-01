import ast
import logging
import argparse
import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import architectures.resnet50_2c
import sys
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')


class ImageSemSegDataset(Dataset):
    def __init__(self, dataset, transforms, size):
        self.dataset = dataset
        self.transforms = transforms
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pil_img = self.dataset[idx]['image']

        sem_seg_img = np.zeros((pil_img.size[1], pil_img.size[0]), dtype=np.uint8)
        for bounding_box in self.dataset[idx]['objects']['bbox']:
            rounded_bbox = [round(bounding_box[i]) for i in range(4)]
            cv2.rectangle(sem_seg_img, rounded_bbox, 1, thickness=-1)

        pil_img = pil_img.resize(self.size)
        img_tsr = self.transforms(pil_img).float()  # (3, H, W)
        sem_seg_img = cv2.resize(sem_seg_img, (img_tsr.shape[2], img_tsr.shape[1]))
        output_tsr = torch.from_numpy(sem_seg_img).float().unsqueeze(0)  # (1, H, W)

        return (img_tsr, output_tsr)

def main(
    outputDirectory,
    resizeSize,
    learningRate,
    weightDecay,
    betas,
    batchSize,
    numberOfEpochs
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    logging.info(f"finetune_FCN_ResNet50.main(): device = {device}")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    dataset = load_dataset("keremberke/license-plate-object-detection", name="full")

    weights = FCN_ResNet50_Weights.DEFAULT
    transforms = weights.transforms(resize_size=None)

    train_dataset = ImageSemSegDataset(dataset['train'], transforms, resizeSize)
    validation_dataset = ImageSemSegDataset(dataset['validation'], transforms, resizeSize)
    test_dataset = ImageSemSegDataset(dataset['test'], transforms, resizeSize)

    # Create data loaders
    train_dataLoader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    validation_dataLoader = DataLoader(validation_dataset, batch_size=batchSize, shuffle=True)

    # Create the neural network
    neural_net = architectures.resnet50_2c.ResNet50_2C()
    neural_net.to(device)

    # Optimization parameters
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, neural_net.parameters()), lr=learningRate,
                           betas=betas,
                           weight_decay=weightDecay)

    lowest_validation_loss = sys.float_info.max
    # Start training
    with open(os.path.join(outputDirectory, "epochLoss.csv"), 'w') as epoch_loss_file:
        epoch_loss_file.write("Epoch,train_loss,validation_loss,is_champion\n")
        for epoch in range(1, numberOfEpochs + 1):
            logging.info("*** Epoch {} ***".format(epoch))
            running_loss = 0.0
            number_of_batches = 0
            neural_net.train()
            for (i, (train_input_tsr, train_target_output_tsr)) in enumerate(train_dataLoader, 0):
                print('.', end='', flush=True)
                train_input_tsr = train_input_tsr.to(device)
                train_target_output_tsr = train_target_output_tsr.to(device)
                optimizer.zero_grad()
                train_output_tsr = neural_net(train_input_tsr)
                loss = criterion(train_output_tsr, train_target_output_tsr)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                number_of_batches += 1
            train_average_loss = running_loss / number_of_batches

            # Validation
            with torch.no_grad():
                neural_net.eval()
                validation_running_loss = 0.0
                number_of_batches = 0
                for (valid_input_tsr, valid_target_output_tsr) in validation_dataLoader:
                    valid_input_tsr = valid_input_tsr.to(device)
                    valid_target_output_tsr = valid_target_output_tsr.to(device)
                    valid_output_tsr = neural_net(valid_input_tsr)
                    loss = criterion(valid_output_tsr, valid_target_output_tsr)
                    validation_running_loss += loss.item()
                    number_of_batches += 1

                validation_average_loss = validation_running_loss / number_of_batches
                is_champion = False


                if validation_average_loss < lowest_validation_loss:
                    lowest_validation_loss = validation_average_loss
                    champion_filepath = os.path.join(outputDirectory,
                                                     f"FCN_ResNet50.pth")
                    torch.save(neural_net.state_dict(), champion_filepath)
                    is_champion = True

                logging.info("Epoch {}; train_loss = {}; validation_loss = {}".format(epoch, train_average_loss,
                                                                                      validation_average_loss))
                if is_champion:
                    logging.info(f" ++++ Champion! ++++")
                epoch_loss_file.write(f"{epoch},{train_average_loss},{validation_average_loss},{is_champion}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_finetune_FCN_ResNet50'",
                        default='./output_finetune_FCN_ResNet50')
    parser.add_argument('--resizeSize', help="The size of the resizing transformation. Default: '(520, 520)'", default='(520, 520)')
    parser.add_argument('--learningRate', help="The learning rate. Default: 0.0001", type=float, default=0.0001)
    parser.add_argument('--weightDecay', help="The weight decay. Default: 0.0001", type=float, default=0.0001)
    parser.add_argument('--betas', help="The optimizer betas. Default: '(0.5, 0.999)'", default='(0.5, 0.999)')
    parser.add_argument('--batchSize', help="The batch size. Default: 16", type=int, default=16)
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 10", type=int, default=10)
    args = parser.parse_args()
    args.resizeSize = ast.literal_eval(args.resizeSize)
    args.betas = ast.literal_eval(args.betas)

    main(
        args.outputDirectory,
        args.resizeSize,
        args.learningRate,
        args.weightDecay,
        args.betas,
        args.batchSize,
        args.numberOfEpochs
    )
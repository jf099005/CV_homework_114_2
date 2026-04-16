# ============================================================================
# File: p2_inference.py
# Date: 2026-03-27
# Author: TA
# Description: Load pre-trained model and perform inference on test set.
# ============================================================================
import os
import sys
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
import json
import config as cfg

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import write_csv


def validate(model, val_loader, device, threshold):
    with torch.no_grad():
        cond_val_correct = 0.0
        val_n_total = 0
        for batch, data in enumerate(val_loader):
            # Data loading. (batch_size, 3, 32, 32), (batch_size)
            images, labels = data['images'].to(device), data['labels'].to(device)
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            pred = model(images)
            pred_n = torch.softmax(pred, dim=1)

            max_probs, pseudo_labels = torch.max(pred_n, dim=1)
            indexes = torch.where(max_probs >= threshold)[0]
            val_n_total += len(indexes)
            cond_val_correct += torch.sum( pseudo_labels[indexes] == labels[indexes] )

    # Print validation result
    cond_val_acc = cond_val_correct / val_n_total
    return cond_val_acc


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        help='mynet or resnet18',
                        type=str,
                        default='resnet18')
    parser.add_argument('--model_path',
                        type = str,
                        default = './checkpoint/resnet18_best.pth')
    parser.add_argument('--output_annotations_path',
                        help='output csv file path',
                        type=str,
                        default='./pseudo_annotations.json')
    
    parser.add_argument('--val_dataset_path', 
                        type = str, 
                        default = None)

    parser.add_argument('--dataset_path',
                        type = str)
    parser.add_argument('--threshold', 
                        type = float, 
                        default = 0.9)
    
    args = parser.parse_args()

    model_type = args.model_type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if model_type == 'mynet':
        model = MyNet()
        model.load_state_dict(torch.load(args.model_path, 
                                         map_location=torch.device('cpu')))
    elif model_type == 'resnet18':
        model = ResNet18()
        model.load_state_dict(torch.load(args.model_path, 
                                         map_location=torch.device('cpu')))
    else:
        raise NameError('Unknown model type')
    model.to(device)

    filenames = []
    filepaths = []
    labels = []
    model.eval()
    if args.val_dataset_path:
        val_loader = get_dataloader(
            args.val_dataset_path,
            batch_size=cfg.batch_size, 
            split='val', 
        )
        cond_val_acc = validate(model=model, val_loader=val_loader, device=device, threshold = args.threshold)
        print(f'threashold {args.threshold}, confidence level: {cond_val_acc:.5f}')
    
    else:
        print("Validation dataset is not given, skip the confidence-level estimation.")

    print("Pseudo-Label generation start.")

    data_loader = get_dataloader(
        args.dataset_path,
        batch_size=cfg.batch_size, split='test',
        return_img_names= True
    )

    with torch.no_grad():
        for batch, data in tqdm(enumerate(data_loader)):
            images = data['images'].to(device)
            img_names = data['image_names']

            pred = model(images)
            pred_n = torch.softmax(pred, dim=1)

            max_probs, pseudo_labels = torch.max(pred_n, dim=1)

            indexes = torch.where(max_probs >= args.threshold)[0]

            for idx in indexes:
                filepaths.append( os.path.join( args.dataset_path, img_names[idx] ) )
                filenames.append(img_names[idx])
                labels.append(pseudo_labels[idx].item())

    with open(args.output_annotations_path, 'w') as f:
        json.dump(
            {
                'filenames': filenames,
                'filepaths': filepaths,
                'labels': labels
            },
            f,
            indent=2
        )
    print(f'Pseudo Label generation ended, totally {len(labels)} samples.')

if __name__ == '__main__':
    main()

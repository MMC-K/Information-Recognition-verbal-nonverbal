##### vit for experiment

import argparse

import os
import gc
import time
import json
import shutil
import logging
import functools
import csv

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import optim
import torch.utils.data

from transformers import AutoTokenizer, ViTFeatureExtractor
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.tensorboard import SummaryWriter
#import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


from data_utils import DatasetForVLAlign


NUM_CLASSES = 7

logger = logging.getLogger(__name__)

def compute(model, batch, loss_fn, args):
    outputs = model(batch['pixel_values'])

    logits = outputs.logits
    target = batch['labels'].to(logits.device)
    loss = loss_fn(logits, target)

    _, predicted_labels = torch.max(logits, dim=1)
    accuracy = torch.mean((predicted_labels == target).float())

    return loss, accuracy


def emotion_eval(model, batch):
    outputs = model(batch['pixel_values'])
    logits = outputs.logits

    #print(logits)

    _, predicted_labels = torch.max(logits, dim=1)

    true_labels = batch['labels'].cpu().numpy()
    predicted_labels = predicted_labels.cpu().numpy()

    return true_labels, predicted_labels

def evaluate(model, eval_loader):
    model.eval()
    total_labels = []
    total_preds = []

    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to('cuda:0')
            labels = batch["labels"].to('cuda:0')
            
            batch = {
                "pixel_values": pixel_values,
                "labels": labels
            }

            labels, preds = emotion_eval(model, batch)
            total_labels.extend(labels)
            total_preds.extend(preds)

    target_names = ['기쁨', '슬픔', '분노', '당황', '불안', '상처', '중립']
    report = classification_report(total_labels, total_preds, target_names=target_names)
    confusion = confusion_matrix(total_labels, total_preds)

    return report, confusion



def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default='/home/jeeyoung/test.json',
                        type=str, help="path to the test data")
    parser.add_argument("--image_root_dir",
                        default=None, type=str)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)
    
    parser.add_argument("--num_classes", default=NUM_CLASSES, type=int,
                        help="number of classes")

    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    
    parser.add_argument("--hf_path",
                        default='./output/vit/weights/',
                        help="path to score huggingface model")
                    

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=8,
                        type=int, help="mini batch size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")
    parser.add_argument("--print_freq", default=1000,
                        type=int, help="print frequency")
    parser.add_argument("--global_steps", default=0,
                        type=int, help="variable for global steps")

    # default settings for training
    parser.add_argument("--epochs", default=1, type=int,
                        help="number of epochs for training")
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")
    parser.add_argument("--save_freq", default=1000,
                        type=int, help="steps to save checkpoint")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-4,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup", default=0.1,
                        type=float, help="warm-up proportion for linear scheduling")
    parser.add_argument("--logit_temperature", default=1.0,
                        type=float, help="temperature for logits")
    parser.add_argument("--label_smoothing", default=0.1,
                        type=float, help="label smoothing for cross entropy")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--off_scheduling", action='store_false',
                        help="off_scheduling")
    parser.add_argument("--max_validation_steps", default=1000, type=int,
                        help="max steps for validation")
    
    # ddp settings for sync
    parser.add_argument("--seed", default=0,
                        type=int, help="seed for torch manual seed")
    parser.add_argument("--deterministic", action='store_true',
                        help="deterministic")
    parser.add_argument("--save_every_epoch", action='store_true',
                        help="save check points on every epochs")
    parser.add_argument("--freeze_lm", action='store_true',
                        help="freeze language model")

    args = parser.parse_args()
    
    # device
    device = torch.device('cuda')

    # deterministic seed
    if args.deterministic:
        torch.manual_seed(args.seed)
        data_seed = args.seed
    else:
        data_seed = torch.randint(9999, (1,), device=device, requires_grad=False)
        data_seed = data_seed.cpu().item()
        logger.info("[rank {}]seed for data: {}".format(0, data_seed))

    # update batch_size per a device
    args.batch_size = int(
        args.batch_size / args.gradient_accumulation_steps)

    model_name = 'google/vit-base-patch16-384'
    model = ViTForImageClassification.from_pretrained(model_name)
    model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
    checkpoint_path = "./output/vit/weights/best_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # get tokenizer
    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)


    # create dataset
    test_dataset = DatasetForVLAlign(
        file_path=args.data_path,
        image_tokenizer=image_tokenizer,
        text_tokenizer=text_tokenizer,
        image_root_dir=args.image_root_dir
    )
    collate_fn = test_dataset.get_collate_fn()

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn)

    model = model.to(device)
    for epoch in range(args.start_epoch, args.epochs):
        eval_report, eval_confusion = evaluate(model, test_loader)

        print("Classification Report:\n", eval_report)
        print("Confusion Matrix:\n", eval_confusion)

        # Identify misclassified samples and save to a CSV file
        misclassified_samples = []
        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                paths = batch["path"]

                batch = {
                    "pixel_values": pixel_values,
                    "labels": labels
                }

                true_labels, predicted_labels = emotion_eval(model, batch)

                for i in range(len(true_labels)):
                    if true_labels[i] != predicted_labels[i]:
                        misclassified_samples.append({
                            "path": paths[i],
                            "true_label": true_labels[i],
                            "predicted_label": predicted_labels[i]
                        })

        if misclassified_samples:
            csv_path = "./output/vit/misclassified_samples.csv"
            with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ["path", "true_label", "predicted_label"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for sample in misclassified_samples:
                    writer.writerow(sample)

            print("Misclassified samples saved to:", csv_path)


if __name__ == "__main__":
    main()

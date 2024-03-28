##### vit for experiment

import argparse

import os
import gc
import time
import json
import shutil
import logging
import functools

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


def create_dir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def create_directory_info(args, create_dir=True):

    model_dir = os.path.join(args.output_dir, "vit")
    if args.dir_suffix is not None:
        model_dir = '_'.join([model_dir, args.dir_suffix])
    weights_dir = os.path.join(model_dir, "weights")
    logs_dir = os.path.join(model_dir, "logs")

    path_info = {
        'model_dir': model_dir,
        'weights_dir': weights_dir,
        'logs_dir': logs_dir,
    }

    if create_dir:
        for k, v in path_info.items():
            create_dir_if_not_exist(v)

    path_info['best_model_path'] = os.path.join(weights_dir, "best_model.pth")
    path_info['ckpt_path'] = os.path.join(weights_dir, "checkpoint.pth")
    return path_info

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_path",
                        default="/home/jeeyoung/train.json", type=str)
    parser.add_argument("--validation_path",
                        default="/home/jeeyoung/valid.json", type=str)
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
    parser.add_argument("--output_dir",
                        default="output", type=str)

    # resume
    parser.add_argument("--resume", default=None, type=str,
                        help="path to checkpoint.")
    parser.add_argument("--hf_path", default=None, type=str,
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

    # create directory and summary logger
    best_acc = 0
    path_info = create_directory_info(args)
    summary_logger = SummaryWriter(path_info["logs_dir"])
    path_info = create_directory_info(args, create_dir=False)
    
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
    model = model.to(device)

    # get optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )

    # get tokenizer
    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    # create dataset
    train_dataset = DatasetForVLAlign(
            file_path=args.train_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer,
            image_root_dir=args.image_root_dir
        )

        
    validation_dataset = DatasetForVLAlign(
            file_path=args.validation_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer,
            image_root_dir=args.image_root_dir
        )

    collate_fn = validation_dataset.get_collate_fn()

    # create data loader
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn)

    validation_loader = DataLoader(validation_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn)
    
    
    # learning rate scheduler
    scheduler = None
    if args.off_scheduling:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            last_epoch=-1,
            steps_per_epoch=int(len(train_loader)/args.gradient_accumulation_steps),
            pct_start=args.warmup,
            anneal_strategy="linear"
        )


    model = model.to(device)    

    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))

                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint and scheduler is not None:
                    if checkpoint['scheduler'] is not None:
                        scheduler.load_state_dict(checkpoint['scheduler'])

                args.start_epoch = checkpoint['epoch']
                if args.resume.endswith('-train'):
                    args.global_steps = checkpoint['global_step']
                    logger.info("=> global_steps '{}'".format(args.global_steps))
                    args.start_epoch-=1

                best_acc = checkpoint['best_acc'] if 'best_acc' in checkpoint else 0
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            elif args.resume.lower()=='true':
                args.resume = path_info['ckpt_path']
                resume()
            elif args.resume.lower()=='best':
                args.resume = path_info['best_model_path']
                resume()
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    
    optimizer.param_groups[0]['capturable'] = True

    # save model as huggingface model
    if args.hf_path:
        if args.hf_path.lower()=='default':
            args.hf_path = os.path.join(path_info["model_dir"], "hf")

        model.module.save_pretrained(args.hf_path)
        logger.info('hf model is saved in {}'.format(args.hf_path))
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        total_preds = []
        total_labels = []
        
        # Training
        train(train_loader, model, optimizer, scheduler, epoch, args, path_info, summary_logger=summary_logger)
        args.global_steps = 0

        # Validation
        scores = validate(validation_loader, model, epoch, args, total_labels, total_preds)

        ckpt_path = os.path.join(path_info["weights_dir"], "ckpt-{}.pth".format(epoch)) if args.save_every_epoch else path_info["ckpt_path"]

        is_best = scores["accuracy"] > best_acc
        if scores["accuracy"] > best_acc:
            best_acc = scores["accuracy"]
            best_epoch = epoch+1
        best_acc = max(scores["accuracy"], best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'scheduler': scheduler.state_dict() if scheduler is not None else scheduler,
        }, is_best, ckpt_path, path_info["best_model_path"])

        summary_logger.add_scalar('eval/loss', scores['loss'], epoch)
        summary_logger.add_scalar('eval/accuracy', scores['accuracy'], epoch)

        print(confusion_matrix(total_labels, total_preds))
        target_name = ['기쁨 (0)', '슬픔 (1)', '분노 (2)', '당황 (3)', '불안 (4)', '상처 (5)', '중립 (6)']
        print(classification_report(total_labels, total_preds, target_names = target_name))

    summary_logger.close()
    print('검증 세트 기준 best accuracy : ', best_acc)
    print('검증 세트 기준 best epoch : ', best_epoch)

def train(train_loader, model, optimizer, scheduler, epoch, args, path_info, summary_logger=None):
    loss_fn = CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    steps_per_epoch = len(train_loader)

    model.train()
    end = time.time()

    optimizer.zero_grad()

    for step_inbatch, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to('cuda:0')
        attention_mask = batch["attention_mask"].to('cuda:0')
        pixel_values = batch["pixel_values"].to('cuda:0')
        labels = batch["labels"].to('cuda:0')
        paths = batch["path"]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }

        loss, accuracy = compute(model, batch, loss_fn, args)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item())
        accuracies.update(accuracy)

        global_step = step_inbatch
        if (global_step) % args.print_freq == 0:
            with torch.no_grad():
                batch_time.update((time.time() - end) / args.print_freq)
                end = time.time()

                accuracy_value = accuracies.avg
                summary_logger.add_scalar('train/loss', losses.avg, epoch)
                summary_logger.add_scalar('train/accuracy', accuracy_value, epoch)

                score_log = "loss\t{:.3f}\t accuracy\t{:.3f}".format(losses.avg, accuracy_value)

                logger.info('-----Training----- \nEpoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'.format(
                                epoch, step_inbatch, steps_per_epoch,
                                args.batch_size / batch_time.val,
                                args.batch_size / batch_time.avg,
                                batch_time=batch_time) + score_log)
                
                # logger.info("Path information:")
                # for path in paths:
                #     logger.info(path)

        if (global_step + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else scheduler,
            }, False, path_info["ckpt_path"] + "-train", path_info["best_model_path"])


def validate(eval_loader, model, epoch, args, total_labels, total_preds):
    # loss function
    loss_fn = CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)

    steps_per_epoch = len(eval_loader)

    # score meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode (for drop out)
    model.eval()

    with torch.no_grad():
        end = time.time()

        for step_inbatch, batch in enumerate(eval_loader):
            input_ids = batch["input_ids"].to('cuda:0')
            attention_mask = batch["attention_mask"].to('cuda:0')
            pixel_values = batch["pixel_values"].to('cuda:0')
            labels = batch["labels"].to('cuda:0')
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "labels": labels
            }

            loss, accuracy = compute(model, batch, loss_fn, args)
            labels, preds = emotion_eval(model, batch)
            total_labels.extend(labels)
            total_preds.extend(preds)

            losses.update(loss.item())

            accuracies.update(accuracy)

            if step_inbatch % args.print_freq == 0:
                batch_time.update((time.time() - end) / min(args.print_freq, step_inbatch + 1))
                end = time.time()

                score_log = "loss\t{:.3f}\t accuracy\t{:.3f}".format(losses.avg, accuracies.avg)

                logger.info('-----Evaluation----- \nEpoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'.format(
                                epoch, step_inbatch, steps_per_epoch,
                                args.batch_size / batch_time.val,
                                args.batch_size / batch_time.avg,
                                batch_time=batch_time) + score_log)

    scores = {
        "loss": losses.avg,
        "accuracy": accuracies.avg
    }
    score_log = "loss\t{:.3f}\t accuracy\t{:.3f}\n".format(scores["loss"], scores["accuracy"])

    logger.info('-----Evaluation----- \nEpoch: [{0}]\t'.format(epoch) + score_log)

    return scores



if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    main()

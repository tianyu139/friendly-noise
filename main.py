import argparse
import logging
import os
import random
import shutil
import time
import datetime
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torchvision

from dataset.tinyimagenet_module import TinyImageNet
from utils import AverageMeter, accuracy
from models.resnet import resnet_picker
from models.alexnet import AlexNet
from models.lenet import LeNet

import pickle
from torchvision import datasets
from torchvision import transforms
from dataset.cifar import get_target, cifar10_mean, cifar10_std
from dataset.poison import PerturbedPoisonedDataset

import matplotlib.pyplot as plt
import copy
from utils.misc import GradualWarmupScheduler

from diff_data_augmentation import RandomTransform
from friendly_noise import generate_friendly_noise, UniformNoise, GaussianNoise, BernoulliNoise

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser(description='Friendly Noise Defense')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'tinyimagenet'],
                        help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'alexnet', 'lenet'],
                        help='dataset name')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of epochs')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--val_freq', default=10, type=int,
                        help='how frequent to run validation')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=int,
                        help='warmup epochs (default: 0)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    # Poison Setting
    parser.add_argument('--clean', action='store_true', help='train with the clean data')
    parser.add_argument("--poisons_path", type=str, help="where are the poisons?")
    parser.add_argument('--dataset_index', type=str, default=None, help='dataset index for naming purposes')
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
    parser.add_argument('--trigger_path', type=str, default=None, help='path to the trigger')
    parser.add_argument("--backdoor", action='store_true', help='whether we are using backdoor attack')
    parser.add_argument('--scenario', default='scratch', choices=('scratch', 'transfer', 'finetune'), help='select the training setting')

    parser.add_argument('--no_augment', action='store_true', help='no augment')

    parser.add_argument('--noise_type', type=str, nargs='*', help='type of noise to apply', default=[], choices=["uniform", "gaussian", "bernoulli", "gaussian_blur", "friendly"])
    parser.add_argument('--noise_eps', type=float, help='strength of noise to apply', default=8)

    parser.add_argument('--friendly_begin_epoch', type=int, help='epoch to start adding friendly noise', default=0)
    parser.add_argument('--friendly_epochs', type=int, help='number of epochs to run friendly noise generation for', default=30)
    parser.add_argument('--friendly_lr', type=float, help='learning rate for friendly noise generation', default=100)
    parser.add_argument('--friendly_mu', type=float, help='weight of magnitude constraint term in friendly noise loss', default=1)
    parser.add_argument('--friendly_clamp', type=float, help='how much to clamp generated friendly noise', default=16)
    parser.add_argument('--friendly_loss', type=str, help='loss to use for friendly noise', default='KL', choices=['MSE', 'KL'])
    parser.add_argument('--save_friendly_noise', action='store_true', help='save friendly noise')
    parser.add_argument('--load_friendly_noise', type=str, help='load friendly noise', default=None)



    args = parser.parse_args()

    global best_acc, transform_train, transform_val

    if args.dataset == 'tinyimagenet':
        crop_size = 64
    else:
        crop_size = 32

    params = dict(source_size=crop_size, target_size=crop_size, shift=8, fliplr=True)
    ap_augment = RandomTransform(**params, mode='bilinear')
    transform_train = []

    if "uniform" in args.noise_type:
        print(f"Adding uniform noise: {args.noise_eps}")
        transform_train.append(UniformNoise(eps=args.noise_eps / 255))
    if "gaussian" in args.noise_type:
        print(f"Adding gaussian noise: {args.noise_eps}")
        transform_train.append(GaussianNoise(eps=args.noise_eps / 255))
    if "bernoulli" in args.noise_type:
        print(f"Adding bernoulli noise: {args.noise_eps}")
        transform_train.append(BernoulliNoise(eps=args.noise_eps / 255))
    if "friendly" in args.noise_type:
        print(f"Using friendly noise")

    transform_train.append(transforms.Normalize(mean=cifar10_mean, std=cifar10_std))
    if args.no_augment:
        pass
    else:
        transform_train.append(ap_augment)
    transform_train = transforms.Compose(transform_train)

    transform_train_noaug = transforms.Compose([
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    def create_model(args):
        if args.arch == 'resnet18':
            model = resnet_picker('ResNet18', 'CIFAR10')
        elif args.arch == 'lenet':
            model = LeNet()
        elif args.arch == 'alexnet':
            model = AlexNet(args.num_classes)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    if args.seed is not None:
        set_seed(args)

    if args.dataset_index is None or args.dataset_index == '':
        dataset_index = args.poisons_path.split('/')[-2]
    else:
        dataset_index = args.dataset_index

    if args.epochs == 200:
        args.steps = [int(args.epochs*1/2), int(args.epochs*3/4)]
    else:
        args.steps = [args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142]

    dir_name = f'{args.arch}-{args.dataset}'
    dir_name += '-clean' if args.clean else f'-{dataset_index}'
    dir_name += f'.{args.seed}-sl-epoch{args.epochs}'
    dir_name += f'-warmup{args.warmup}' if args.warmup > 0 else ''
    dir_name += f'-{args.scenario}' if args.scenario != 'scratch' else ''
    dir_name += f'-noaug' if args.no_augment else ''
    dir_name += f'-{"-".join(args.noise_type)}' if len(args.noise_type) != 0 else ''
    dir_name += f'-p{args.friendly_begin_epoch}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_{args.friendly_epochs}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_{args.friendly_lr}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_{args.friendly_mu}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_clp{args.friendly_clamp}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_loss-{args.friendly_loss}' if 'friendly' in args.noise_type else ''
    args.out = os.path.join(args.out, dir_name)


    os.makedirs(args.out, exist_ok=True)

    # write and save training log
    logging.basicConfig(
        filename=f"{args.out}/output.log",
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}")

    logger.info(dict(args._get_kwargs()))

    if args.dataset == 'cifar10':
        args.num_classes = 10
        base_dataset = datasets.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_val, download=False)

    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        base_dataset = TinyImageNet('./data/tiny-imagenet-200')
        test_dataset = TinyImageNet('./data/tiny-imagenet-200', split='val', transform=transform_val)

    args.train_size = len(base_dataset)

    if args.clean:
        train_dataset = PerturbedPoisonedDataset(
            trainset=base_dataset,
            indices=np.array(range(len(base_dataset))),
            transform=transform_train,
            return_index=True,
            size=args.train_size)

        noaug_train_dataset = PerturbedPoisonedDataset(
            trainset=base_dataset,
            indices=np.array(range(len(base_dataset))),
            transform=transform_train_noaug,
            return_index=True,
            size=args.train_size)

        poison_indices = []
        poison_tuples = []
        target_class = -1
        poisoned_label = -1
        target_img = None

    elif args.poisons_path is not None:
        # load the poisons and their indices within the training set from pickled files
        if os.path.isfile(args.poisons_path):
            with open(args.poisons_path, "rb") as handle:
                print(f"Loading MetaPoison datasets...")
                poison_data = pickle.load(handle)
                to_pil = transforms.ToPILImage()
                base_dataset.data = np.uint8(poison_data['xtrain'])
                base_dataset.targets = poison_data['ytrain']
                target_img = transform_val(to_pil(np.uint8(poison_data['xtarget'][0])))
                target_class = poison_data['ytarget'][0]
                poisoned_label = poison_data['ytargetadv'][0]
                poison_indices = np.array(range(5000*poisoned_label, 5000*poisoned_label+500))
                poison_tuples = []
                for i in poison_indices:
                    poison_tuples.append((to_pil(np.uint8(poison_data['xtrain'][i])), poison_data['ytrain'][i]))
        else:
            with open(os.path.join(args.poisons_path, "poisons.pickle"), "rb") as handle:
                poison_tuples = pickle.load(handle)
                logger.info(f"{len(poison_tuples)} poisons in this trial.")
                poisoned_label = poison_tuples[0][1]
            with open(os.path.join(args.poisons_path, "base_indices.pickle"), "rb") as handle:
                poison_indices = pickle.load(handle)
            target_img, target_class = get_target(args, transform_val)

        train_dataset = PerturbedPoisonedDataset(
            trainset=base_dataset,
            indices=np.array(range(len(base_dataset))),
            poison_instances=poison_tuples,
            poison_indices=poison_indices,
            transform=transform_train,
            return_index=True,
            size=args.train_size)

        noaug_train_dataset = PerturbedPoisonedDataset(
            trainset=base_dataset,
            indices=np.array(range(len(base_dataset))),
            poison_instances=poison_tuples,
            poison_indices=poison_indices,
            transform=transform_train_noaug,
            return_index=True,
            size=args.train_size)

    else:
        raise ValueError('poisons path cannot be empty')


    model = create_model(args)
    model.to(args.device)

    logger.info(f"Target class {target_class}; Poisoned label: {poisoned_label}")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)

    noaug_train_loader = DataLoader(
        noaug_train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.scenario != 'scratch':
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if args.scenario == 'transfer':
            logger.info("==> Freezing the feature representation..")
            for param in model.parameters():
                param.requires_grad = False
        else:
            logger.info("==> Decreasing the learning rate for fine-tuning..")
            args.lr = 1e-4
        logger.info("==> Reinitializing the classifier..")
        num_ftrs = model.linear.in_features
        model.linear = nn.Linear(num_ftrs, args.num_classes).to(args.device)  # requires grad by default

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steps)
    if args.warmup > 0:
        logger.info('Warm start learning rate')
        lr_scheduler_f = GradualWarmupScheduler(optimizer, 1.0, args.warmup, scheduler)
    else:
        logger.info('No Warm start')
        lr_scheduler_f = scheduler

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    args.start_epoch = 0

    if args.resume and (args.scenario == 'scratch'):
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.load_friendly_noise:
        logger.info(f"Loading friendly noise from {args.load_friendly_noise}...")
        perturbs = np.load(args.load_friendly_noise)
        train_loader.dataset.set_perturbations(torch.Tensor(friendly))
        logger.info(f"Friendly noise loaded!")


    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")

    model.zero_grad()
    model_to_save = model.module if hasattr(model, "module") else model
    save_checkpoint({
        'epoch': 0,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, is_best=False, checkpoint=args.out, filename='init.pth.tar')


    train(args, train_loader, noaug_train_loader, test_loader, model, optimizer, lr_scheduler_f, target_img, target_class, poisoned_label, train_dataset, loss_fn, poison_indices, base_dataset, poison_tuples)

def train(args, trainloader, noaug_trainloader, test_loader, model, optimizer, scheduler, target_img, target_class, poisoned_label, train_dataset, loss_fn, poison_indices, base_dataset, poison_tuples):
    global best_acc
    test_accs = []
    poison_accs = []
    cluster = []
    time_start = time.time()
    end = time.time()

    model.train()
    N = args.train_size
    weights = torch.ones(N)
    times_selected = torch.zeros(N)

    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"***** Epoch {epoch} *****")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        num_poison_selected = torch.tensor(0)

        args.eval_step = len(trainloader)
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        model.train()

        if 'friendly' in args.noise_type and epoch == args.friendly_begin_epoch:
            logger.info(f"Generating friendly noise: epochs={args.friendly_epochs}  mu={args.friendly_mu} lr={args.friendly_lr} loss={args.friendly_loss}")
            out = generate_friendly_noise(
                model,
                noaug_trainloader,
                args.device,
                friendly_epochs=args.friendly_epochs,
                mu=args.friendly_mu,
                friendly_lr=args.friendly_lr,
                clamp_min=-args.friendly_clamp / 255,
                clamp_max=args.friendly_clamp / 255,
                return_preds=args.save_friendly_noise,
                loss_fn=args.friendly_loss)
            model.zero_grad()
            model.train()

            if args.save_friendly_noise:
                friendly_noise, original_preds = out
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'poison_acc': p_acc,
                    'best_poison_acc': best_p_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'times_selected': times_selected,
                }, is_best=False, checkpoint=args.out, filename='model-friendlygen.pth.tar')
                friendly_path = os.path.join(args.out, 'friendly-friendlygen.npy')
                pred_path = os.path.join(args.out, 'original-preds-friendlygen.npy')
                np.save(friendly_path, friendly_noise.numpy())
                np.save(pred_path, original_preds.numpy())
                logger.info(f"Saved friendly noise to {friendly_path}")
                logger.info(f"Saved original predictions to {pred_path}")
            else:
                friendly_noise = out

            trainloader.dataset.set_perturbations(friendly_noise)
            logger.info(f"Friendly noise stats:  Max: {torch.max(friendly_noise)}  Min: {torch.min(friendly_noise)}  Mean (abs): {torch.mean(torch.abs(friendly_noise))}  Mean: {torch.mean(friendly_noise)}")

        for batch_idx, batch_input in enumerate(trainloader):
            input, targets_u_gt, p, index = batch_input
            targets_u_gt = targets_u_gt.long()
            num_poison_selected += torch.sum(p)

            data_time.update(time.time() - end)
            logits_u_w = model(input.to(args.device))
            pseudo_label = torch.softmax(logits_u_w, dim=-1)
            probs_u, targets_u = torch.sort(pseudo_label, dim=-1, descending=True)
            max_probs, targets_u = probs_u[:, 0], targets_u[:, 0]


            loss = loss_fn(logits_u_w, targets_u_gt.to(args.device))
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item())
            optimizer.step()
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg))
                p_bar.update()

            times_selected[index] += 1

        if not args.no_progress:
            p_bar.close()

        scheduler.step()

        if epoch % args.val_freq != 0 and epoch != args.epochs - 1:
            continue

        test_model = model

        # test poisoning success
        test_model.eval()
        if args.clean:
            p_acc = -1
            t_acc = -1
            pass
        elif args.backdoor:
            p_accs = []
            t_accs = []
            for t in target_img:
                target_conf = torch.softmax(test_model(t.unsqueeze(0).to(args.device)), dim=-1)
                target_pred = target_conf.max(1)[1].item()
                p_acc = (target_pred == poisoned_label)
                t_acc = (target_pred == target_class)

                p_accs.append(p_acc)
                t_accs.append(t_acc)

            p_acc = np.mean(p_accs)
            t_acc = np.mean(t_accs)
        else:
            target_conf = torch.softmax(test_model(target_img.unsqueeze(0).to(args.device)), dim=-1)
            target_pred = target_conf.max(1)[1].item()
            p_acc = (target_pred == poisoned_label)
            t_acc = (target_pred == target_class)

        test_loss, test_acc = test(args, test_loader, test_model, loss_fn, p_acc)

        print(f"Poison acc: {p_acc}")

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            best_p_acc = p_acc

        model_to_save = model.module if hasattr(model, "module") else model
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'poison_acc': p_acc,
            'best_poison_acc': best_p_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'times_selected': times_selected,
        }, is_best, args.out)

        test_accs.append(test_acc)
        poison_accs.append(p_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}'.format(
            np.mean(test_accs[-20:])))
        logger.info(f'Target acc: {t_acc}')
        logger.info(f'Poison acc: {p_acc}\n')

    time_end = time.time()
    # Save to csv output
    save_path = os.path.join(args.out, 'result.csv')
    save_dict = {
        'epoch': epoch + 1,
        'acc': test_acc,
        'best_acc': best_acc,
        'poison_acc': p_acc,
        'best_poison_acc': best_p_acc,
        'mean_poison_acc': np.mean(poison_accs),
        'runtime': str(datetime.timedelta(seconds=time_end - time_start)).replace(',', ''),
    }
    save_dict = {**save_dict, **(vars(args))}

    print(f"Saving final results to {save_path}")
    df = pd.DataFrame.from_dict([save_dict])
    df.to_csv(save_path, mode='a', index=False)


def test(args, test_loader, model, loss_fn, p_acc):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets).mean()

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. P_acc: {p_acc:.3f} Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    p_acc=p_acc,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("poison acc: {:.2f}".format(p_acc))
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()

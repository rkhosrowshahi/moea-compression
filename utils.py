"""
    setup model and datasets
"""


import copy
import os
import random

# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import sys
import time

import numpy as np
import torch
import torchvision
from dataset import *
from dataset import TinyImageNet
from imagenet import prepare_data
from models import *
from torchvision import transforms
from tqdm import tqdm

__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
]


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def save_checkpoint(
    state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
        )


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


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


def dataset_convert_to_train(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = train_transform
    dataset.train = False


def get_balanced_subset(dataset, num_samples, num_classes):
    samples_per_class = num_samples // num_classes
    if num_samples % num_classes > 0:
        samples_per_class += 1
    # Create an empty list to store the balanced dataset
    balanced_indices = []
    # Randomly select samples from each class for the training dataset
    for i in range(num_classes):
        class_indices = np.where(np.array(dataset.targets) == i)[0]
        selected_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )
        balanced_indices.append(selected_indices)
    return np.asarray(balanced_indices).astype(int)

def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def setup_model_dataset(args):
    checkpoint = None
    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_full_loader, val_loader, _ = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)
            
        setup_seed(args.train_seed)
        checkpoint = load_checkpoint(device="cuda", save_path=f"./checkpoints/cifar10/{args.arch}/", pruning=0, filename="model_SA_best.pth.tar")
        if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "svhn":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
        )
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = svhn_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)
        
        checkpoint = load_checkpoint(device="cuda", save_path=f"./checkpoints/svhn/{args.arch}/", pruning=0, filename="model_SA_best.pth.tar")
        if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_full_loader, val_loader, _ = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)
        checkpoint = load_checkpoint(device="cuda", save_path=f"./checkpoints/cifar100/{args.arch}/", pruning=0, filename="model_SA_best.pth.tar")
        if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "TinyImagenet":
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_full_loader, val_loader, test_loader = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        # train_full_loader, val_loader, test_loader =None, None,None
        marked_loader, _, _ = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "imagenet":
        classes = 1000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # train_ys = torch.load(args.train_y_file)
        # val_ys = torch.load(args.val_y_file)
        model = model_dict[args.arch](num_classes=classes, imagenet=True, pretrained=True, normalization=normalization)
        # from torchvision.models import resnet18, resnet34, resnet50, resnet101
        # model = resnet18(weights="DEFAULT")
        # # train_subset_indices = torch.ones_like(train_ys)
        # # val_subset_indices = torch.ones_like(val_ys)
        # # train_subset_indices[train_ys] = 0
        # # val_subset_indices[val_ys == args.class_to_replace] = 0
        # loaders = prepare_data(
        #         dataset="imagenet",
        #         batch_size=args.batch_size,
        # )
        # retain_loader = loaders["train"]
        # val_loader = loaders["val"]
        num_classes = 1000
        val_num_samples = 10000
        test_num_samples = 40000
        transform_test = transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )
        testset = torchvision.datasets.ImageNet(
            root="./data/imagenet-1000",
            split="val",
            transform=transform_test,
        )
        if os.path.exists("./data/imagenet-1000/imagenet_data_split.npz"):
            c = np.load("./data/imagenet-1000/imagenet_data_split.npz")
            val_indices = c["val"]
            test_indices = c["test"]
        else:
            balanced_indices = get_balanced_subset(
                testset, val_num_samples + test_num_samples, num_classes
            )
            val_indices = balanced_indices[
                :, : (val_num_samples // num_classes)
            ].flatten()
            np.random.shuffle(val_indices)
            test_indices = balanced_indices[
                :, (val_num_samples // num_classes) :
            ].flatten()
            # test_indices = balanced_indices.flatten()
            np.random.shuffle(test_indices)
            np.savez(
                "./data/imagenet-1000/imagenet_data_split.npz",
                val=val_indices,
                test=test_indices,
            )
        # test_set = testset
        # val_set = testset
        val_set = Subset(testset, val_indices)
        test_set = Subset(testset, test_indices)
        val_loader = DataLoader(val_set, batch_size=128, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)
        return model, val_loader, val_loader, test_loader, None

    elif args.dataset == "cifar100_no_val":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    elif args.dataset == "cifar10_no_val":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    else:
        raise ValueError("Dataset not supprot yet !")
    # import pdb;pdb.set_trace()

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    checkpoint = load_checkpoint(device="cuda", save_path=f"./checkpoints/cifar10/{args.arch}/", pruning=0, filename="model_SA_best.pth.tar")
    if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])

    model.normalize = normalization
    return model, train_set_loader, val_loader, test_loader, None


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open("stop_{}.sh".format(dir), "w")
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, "w")
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
    )


def get_unlearn_loader(marked_loader, args):
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(
        forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(
        retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    print("datasets length: ", len(forget_dataset), len(retain_dataset))
    return forget_loader, retain_loader


def get_poisoned_loader(poison_loader, unpoison_loader, test_loader, poison_func, args):
    poison_dataset = copy.deepcopy(poison_loader.dataset)
    poison_test_dataset = copy.deepcopy(test_loader.dataset)

    poison_dataset.data, poison_dataset.targets = poison_func(
        poison_dataset.data, poison_dataset.targets
    )
    poison_test_dataset.data, poison_test_dataset.targets = poison_func(
        poison_test_dataset.data, poison_test_dataset.targets
    )

    full_dataset = torch.utils.data.ConcatDataset(
        [unpoison_loader.dataset, poison_dataset]
    )

    poisoned_loader = get_loader_from_dataset(
        poison_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    poisoned_full_loader = get_loader_from_dataset(
        full_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    poisoned_test_loader = get_loader_from_dataset(
        poison_test_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )

    return poisoned_loader, unpoison_loader, poisoned_full_loader, poisoned_test_loader



def load_params(parameters, model, n_params=0, codebook=None, device='cuda'):
    if codebook is not None:
        parameters = unblocker(codebook, n_params, parameters)

    counted_params = 0
    for layer in model.parameters():
        # if not 'norm' in name:
            layer_size = layer.size()
            layer_size_numel = layer_size.numel()
            layer.data = torch.from_numpy(
                parameters[counted_params : layer_size_numel + counted_params].reshape(
                    layer_size
                )
            ).to(device=device, dtype=torch.float32)
            counted_params += layer.size().numel()
    
    return model

def var_load_params(parameters, model, n_params=0, codebook=None, device='cuda'):
    if codebook is not None:
        parameters = var_unblocker(codebook, n_params, parameters)

    counted_params = 0
    for layer in model.parameters():
        # if not 'norm' in name:
            layer_size = layer.size()
            layer_size_numel = layer_size.numel()
            layer.data = torch.from_numpy(
                parameters[counted_params : layer_size_numel + counted_params].reshape(
                    layer_size
                )
            ).to(device=device, dtype=torch.float32)
            counted_params += layer.size().numel()
    
    return model

def get_params(model):
    return np.concatenate([p.flatten().detach().cpu().numpy() for p in model.parameters()])

def total_params(model):
    return np.sum([p.numel() for p in model.parameters()])

def compute_l2_norm(x):
    return np.nanmean(x*x, axis=1)

def compute_ranks(fitness):
    ranks = np.zeros(len(fitness))
    # ranks[fitness.argsort()] = np.arange(len(fitness))
    ranks[fitness.argsort()] = np.linspace(0, 1, len(fitness))
    return ranks

def centered_rank_trafo(fitness):
    y = compute_ranks(fitness)
    return y

def two_obj_ranking(fitness):
    ranks = np.argsort(fitness, axis=0)
    avg_ranks = np.mean(ranks, axis=1)
    y_ranks = avg_ranks.argsort()
    y = np.zeros(fitness.shape)
    # y[avg_ranks.argsort()] = np.array([np.linspace(0, 1, len(fitness)), np.linspace(0, 1, len(fitness))]).T
    x = np.linspace(0, 1, len(fitness))
    # Combine the two halves
    output_array = np.concatenate((x[:int(len(fitness)/2)], x[int(len(fitness)/2):][::-1]))
    y[y_ranks] = output_array
    return y

def fitness_reshaper(X, fitness, w_decay=0.1, norm=True, maximize=True):
    if maximize == True:
        fitness = 1 - fitness
    
    fitness[np.isnan(fitness)] = np.inf

    if w_decay > 0 :
        l2_fit_red = w_decay * compute_l2_norm(X)
        fitness += l2_fit_red

    if norm == True:
        fitness = centered_rank_trafo(fitness)

    return fitness

def two_obj_reshaper(X, fitness, w_decay=0.1, norm=True, maximize=True):
    fitness[np.isnan(fitness)] = np.inf
    if w_decay > 0:
        l2_fit_red = w_decay * compute_l2_norm(X)
        fitness += np.repeat(l2_fit_red, fitness.shape[1]).reshape(-1, fitness.shape[1])
    
    if norm == True:
        fitness = two_obj_ranking(fitness)

    return fitness

def initialization(N, x0, codebook=None):
    init_pop = None
    if codebook:
        BD = len(codebook)

        init_pop = np.random.normal(loc=blocker(x0, codebook), scale=0.001, size=(N, BD))
        
        # for i in range(N):
        #     d=0
        #     for idx, block in codebook.items():
        #         init_pop[i, d] = np.random.uniform(low=x0[block].min(), high=x0[block].max())
        #         d+=1
    
    else:
        init_pop = np.random.normal(loc=x0, scale=0.01, size=(N, len(x0)))
    
    init_pop[0] = blocker(x0, codebook)

    return init_pop

def var_initialization(N, x0, codebook=None):
    init_pop = None
    if codebook:
        BD = len(codebook) * 2

        init_pop = np.zeros((N, BD))
        init_pop = np.random.normal(loc=var_blocker(x0, codebook), scale=0.01, size=(N, BD))
        
        # for i in range(N):
        #     d=0
        #     for idx, block in codebook.items():
        #         init_pop[i, d] = np.random.uniform(low=x0[block].min(), high=x0[block].max())
        #         init_pop[i, d+1] = np.random.uniform(low=0, high=x0[block].std())
        #         d+=2
    
    else:
        init_pop = np.random.normal(loc=x0, scale=0.01, size=(N, len(x0)))
    
    init_pop[0] = var_blocker(x0, codebook)

    return init_pop

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def unblocker(codebook, orig_dims, blocked_params, verbose=False):

    unblocked_params = np.zeros(orig_dims)
    block_idx = 0
    for idx, indices in tqdm(
        codebook.items(),
        desc=f"Unblocking D= {len(blocked_params)} ==> {orig_dims}",
        disable=not verbose,
    ):  
        if len(indices) < 1:
            print(len(indices))
        unblocked_params[indices] = np.full(len(indices), blocked_params[block_idx])
        block_idx += 1
    return unblocked_params

def var_unblocker(codebook, orig_dims, blocked_params, verbose=False):

    unblocked_params = np.zeros(orig_dims)
    block_idx = 0
    for idx, indices in tqdm(
        codebook.items(),
        desc=f"Unblocking D= {len(blocked_params)} ==> {orig_dims}",
        disable=not verbose,
    ):  
        if len(indices) < 1:
            print(len(indices))
        unblocked_params[indices] = np.random.normal(blocked_params[block_idx], abs(blocked_params[block_idx+1]), size=len(indices))
        
        block_idx += 2
    return unblocked_params

def blocker(params, codebook):
    blocked_params = []
    for block_idx, indices in (codebook).items():
        blocked_params.append(params[indices].mean())

    return np.array(blocked_params)

def var_blocker(params, codebook):
    blocked_params = []
    for block_idx, indices in (codebook).items():
        blocked_params.append(params[indices].mean())
        blocked_params.append(0)

    return np.array(blocked_params)


def evaluate(model, data_loader, eval=False, device="cuda"):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if total >= 10000:
            #     break
            if not eval:
                break

    accuracy = correct / total
    return accuracy
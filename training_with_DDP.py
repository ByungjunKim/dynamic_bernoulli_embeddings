# Needed
import re
from tqdm import tqdm

tqdm.pandas()
from gensim.corpora import Dictionary
from nltk import word_tokenize as nltk_word_tokenize

import gc
gc.collect()

# Distributed Data Parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

cudnn.benchmark = True

# added lib
import os
import argparse
import pickle
import random

# original lib
"""Functions for building the training loop"""
import numpy as np
import pandas as pd
import torch

from dynamic_bernoulli_embeddings.embeddings import DynamicBernoulliEmbeddingModel
from dynamic_bernoulli_embeddings.preprocessing import Data, DBEDataloader

def rsave(x, y, rank=0):
    if rank==0: torch.save(x, y)

def rprint(msg, rank=0):
    if rank==0: print(msg)

def _bad_word(word):
    if len(word) < 2:
        return True
    if any(c.isdigit() for c in word):
        return True
    if "/" in word:
        return True
    return False

def word_tokenize(text):
    text = re.sub(r"co-operation", "cooperation", text)
    text = re.sub(r"-", " ", text)
    words = [w.lower().strip("'.") for w in nltk_word_tokenize(text)]
    words = [w for w in words if not _bad_word(w)]
    return words


# For the necessary environment of DDP
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# For the destructor of DDP similar to deallocation in C++
def ddp_clean():
    dist.destroy_process_group()


def train_model_with_DDP(
    rank,  # local rank for representing GPU index
    args,  # arguments
    ngpus_per_node,  # the number of gpu
    dataset_dict,  # dataset dictionary
    **kwargs,
):
    # setup ddp process
    if args.distributed: ddp_setup(rank, ngpus_per_node)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # generating dataset
    train_dataset = DBEDataloader(dataset_dict['data_class'])
    val_dataset = DBEDataloader(dataset_dict['data_val_class'])
    if args.distributed: train_sampler = DistributedSampler(train_dataset, shuffle=True)
    if args.distributed: val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # dataset loader
    train_loader = DataLoader(train_dataset, args.number_of_data_per_batch, shuffle=False if args.distributed else True,
                              num_workers=args.num_workers, sampler=train_sampler if args.distributed else None)
    test_loader = DataLoader(val_dataset, args.number_of_data_per_batch, shuffle=False,
                             num_workers=args.num_workers, sampler=val_sampler if args.distributed else None)

    # Build model.
    model = DynamicBernoulliEmbeddingModel(
        len(dataset_dict['data_class'].dictionary),
        dataset_dict['data_class'].T,
        dataset_dict['data_class'].m_t,
        dataset_dict['data_class'].dictionary,
        dataset_dict['data_class'].unigram_logits.cuda(),
        **kwargs,
    ).cuda()

    # for wrapping DDP
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[rank], output_device=[rank])
        model = model.module

    # Training loop.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_history = []
    for i in range(args.epochs + 1):

        # for shuffle
        if args.distributed: train_sampler.set_epoch(i)

        # Initialize weights from the epoch 0 "burn in" period and reset the optimizer.
        if i == 1:
            with torch.no_grad():
                model.rho.weight = torch.nn.Parameter(
                    model.rho.weight[: model.V].repeat((model.T, 1))
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        pbar.set_description(f"Epoch {i}")
        for j, (targets, contexts, times) in pbar:
            model.train()
            model.zero_grad()
            # The first epoch ignores time for initializing weights.
            if i == 0:
                times = torch.zeros_like(times)
            loss, L_pos, L_neg, L_prior = model(targets, times, contexts, dynamic=i > 0)
            loss.backward()
            optimizer.step()

            # Validation.
            L_pos_val = None
            if args.val_ratio is not None and i > 0 and j % args.val_step == 0:
                L_pos_val = 0
                model.eval()
                for val_targets, val_contexts, val_times in test_loader:
                    _, L_pos_val_batch, _, _ = model(
                        val_targets, val_times, val_contexts, validate=True
                    )
                    L_pos_val += L_pos_val_batch.item()

            # Collect loss history. Ignore the initialization epoch 0.
            if i > 0:
                batch_loss = (
                    loss.item(),
                    L_pos.item(),
                    L_neg.item(),
                    L_prior.item() if L_prior else None,
                    L_pos_val,
                )
                loss_history.append(batch_loss)

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

    loss_history = pd.DataFrame(
        loss_history, columns=["loss", "l_pos", "l_neg", "l_prior", "l_pos_val"]
    )

    # save model checkpoint
    rsave({'model': model.state_dict(),
            'loss_history': loss_history}, os.path.join(args.save_path, 'model.pt'), rank)
    rprint('CHECKPOINT FILE IS SAVED SUCCESSFULLY!', rank)

    # clean ddp process
    if args.distributed: ddp_clean()


if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dataset-path', default="data/un-general-debates.csv", type=str)
    parser.add_argument('--save-path', default="checkpoint/", type=str)
    parser.add_argument('--total_batch', default=300, type=int)
    parser.add_argument('--number_of_data_per_batch', default=128, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--val_step', default=100, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float, help='please 0 to 1')
    parser.add_argument('--num_workers', default=0, type=int, help='0 is appropriate in this work')
    args = parser.parse_args()

    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)

    # cuda visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # random seed for making consistent validation set
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.isfile('data/data.pkl') and args.distributed:

        # dataset loader
        dataset = pd.read_csv(args.dataset_path)
        dataset["bow"] = dataset.text.progress_apply(word_tokenize)
        dataset["time"] = dataset.year - dataset.year.min()

        # dictionary loader
        dictionary = Dictionary(dataset.bow)
        dictionary.filter_extremes(no_below=15, no_above=1.)
        dictionary.compactify()
        print(len(dictionary))

        # dict
        data = {'dataset': dataset, 'dictionary': dictionary}

        # save
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        exit(0)

    elif os.path.isfile('data/data.pkl'):
        # load
        with open('data/data.pkl', 'rb') as f:
            data = pickle.load(f)

    else:
        raise Exception('only works (File non-exist & False Distributed) or (File exist)')

    # Create a validation set and dataset.
    if not os.path.isfile(f'data/dataset_{args.val_ratio}.pkl'):
        validation_mask = np.repeat(False, data['dataset'].shape[0])
        if args.val_ratio is not None:
            assert 0 < args.val_ratio < 1
            validation_mask = np.random.permutation(data['dataset'].shape[0]) / data['dataset'].shape[0] < args.val_ratio
        data_class = Data(df=data['dataset'][~validation_mask], dictionary=data['dictionary'].token2id,
                          m=args.total_batch)
        data_val_class = Data(df=data['dataset'][validation_mask], dictionary=data['dictionary'].token2id,
                              m=args.total_batch)

        # save
        with open(f'data/dataset_{args.val_ratio}.pkl', 'wb') as f:
            pickle.dump({'data_class': data_class, 'data_val_class': data_val_class}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        exit(0)

    else:
        # load
        with open(f'data/dataset_{args.val_ratio}.pkl', 'rb') as f:
            dataset_dict = pickle.load(f)

    if args.distributed:
        # multiprocess spawn
        mp.spawn(train_model_with_DDP,
                 args=(args, ngpus_per_node, dataset_dict,),
                 nprocs=ngpus_per_node,
                 join=True)
    else:
        # first gpu index is activated once there are several gpu in args.gpu
        train_model_with_DDP(rank=0, args=args, ngpus_per_node=1, dataset_dict=dataset_dict)

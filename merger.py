import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune
import copy
from collections import OrderedDict
from tqdm import tqdm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.indicators.hv import HV
from compression import huffman_encoding
from compression.huffman_encoding import HuffmanEncode
from scipy.sparse import csr_array
import arg_parser
# from models.ResNet import resnet18
from compression.pruning import range_prune
from compression.quantization import get_compression, get_model_params, compression, merge_bins_center_to_end, merge_bins_left_to_right
import utils
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use(['science', 'no-latex'])

float32_bits = 32

args = arg_parser.parse_args()
# args.dataset = 'cifar10'
# args.seed = 6
args.model_path = f"./checkpoints/{args.dataset}/{args.arch}/0model_SA_best.pth.tar"
# args.save_dir = f"./logs/cifar100/resnet18/seed{args.seed}/"
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.save_dir+'/history/data', exist_ok=True)
os.makedirs(args.save_dir+'/history/figures', exist_ok=True)

# Define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
if args.gpu == -1:
    device = "cpu"
else:
    device = f"cuda:{args.gpu}"
(
    model,
    train_loader_full,
    val_loader,
    test_loader,
    marked_loader,
    ) = utils.setup_model_dataset(args)
model.to(device)
# checkpoint = torch.load(args.model_path, map_location=device)
# if "state_dict" in checkpoint.keys():
#             checkpoint = checkpoint["state_dict"]
# model.load_state_dict(checkpoint, strict=False)


def size_of_model(model, pruned=False, quantized=False, mask=None):
    name = "weight"
    idx_bits = 5
    compression_dict = {
        n: get_compression(m, name, idx_bits=idx_bits, huffman_encoding=huffman_encoding) for
        n, m in model.named_modules() if
        getattr(m, name, None) is not None}

    for name, (n, d) in compression_dict.items():
        cr = n / d
        # log.info(f"  Layer {name}: compression rate {1 / cr:.2%} ({cr:.1f}X) ")
    n, d = zip(*compression_dict.values())
    total_params = sum(n)
    total_d = sum(d)
    # cr = total_params / total_d
    return total_d
    # log.info(f"Total compression rate: {1 / cr:.2%} ({cr:.1f}X) ")
    

def prune_and_ub_initialization(NP=100, lb_alpha=0, ub_beta=0, problem=None):
    # min_val = params.min()
    # max_val = params.max()
    lb, ub = problem.xl, problem.xu
    lb_K, ub_K = lb[0], ub[0]
    X = np.zeros((NP, 1))
    K = np.linspace(ub_K, lb_K, num=NP).astype(int)
    X = np.column_stack([K])
    return X

def plot_pf(step, pf_F, args):  
            with plt.style.context(['science', 'no-latex']):
                plt.figure(figsize=(5,3))
                plt.scatter(-1 * pf_F[:, 1], 100 * (1 - pf_F[:, 0]))
                plt.xlabel('CR')
                plt.ylabel('Accuracy (%)')
                plt.title(f'Pareto Front (step {step}/100)')
                plt.grid(True)
                plt.savefig(args.save_dir + f'/history/figures/step_{step}.pdf')
                plt.close()
                # plt.show()

def plot_histogram(centers, codebook, state="pre", args=None):
    h = np.zeros(len(centers))
    # print(codebook, centers)
    for i, row in enumerate(codebook.values()):
        h[i] = len(row)
    # print(h.max())
    sorted_range = np.argsort(centers)
    plt.bar(x=centers[sorted_range], height=h[sorted_range], width=0.01)
    plt.yscale('log')
    plt.savefig(args.save_dir + f"/checkpoints/{args.merge_method}/figures/{len(centers)}ws_{state}_merge.pdf")
    plt.close()

if __name__ == "__main__":
    params = get_model_params(model)
    total_params = len(params)
    # pretrained_val_score = utils.evaluate(model, val_loader, eval=True, device=device)
    # print(pretrained_val_score)
    total_model_size = size_of_model(model)
    print(total_params, total_model_size)
    if args.merge:
        print("merge enabled")
        if args.merge_method == "center_to_end":
            merge_bins = merge_bins_center_to_end
        elif args.merge_method == "left_to_right":
            merge_bins = merge_bins_left_to_right

    pf_X = []
    pf_df = None
    
    if os.path.exists(args.save_dir + "results.pth.tar"):
        pf_df = pd.read_csv(args.save_dir + "pf_summary.csv")
        checkpoint = torch.load(args.save_dir + "results.pth.tar")
        os.makedirs(args.save_dir+f'/checkpoints/{args.merge_method}/figures', exist_ok=True)
        # os.makedirs(args.save_dir+f'/checkpoints/{args.merge_method}/merged_models', exist_ok=True)
        pf_X = checkpoint['pf_X']
        # pf_F = checkpoint['pf_F']

    n_pf = len(pf_X) 
    pf_val = np.zeros(n_pf)
    pf_pre_size = np.zeros(n_pf)
    pf_pre_cr = np.zeros(n_pf)
    if 'val_score' in pf_df.columns:
        pf_val = pf_df['val_score'].to_numpy()
        pf_pre_size = pf_df['size'].to_numpy()
    else:                                   
        for i in range(n_pf):
            compressed_model, centers, bin_indices, codebook, n_not_pruned = compression(pf_X[i], model, args)
            pf_val[i] = utils.evaluate(compressed_model, val_loader, eval=True, device=device)
            pf_pre_size[i] = size_of_model(compressed_model)
            pf_pre_cr[i] = total_model_size / pf_pre_size[i]
        pf_df['val_score'] = pf_val
        pf_df['CR'] = pf_pre_cr
        pf_df['size'] = pf_pre_size
        pf_df.to_csv(args.save_dir + f"/pf_summary.csv")

    selected_indices = np.where(pf_df['test_score'] >= 0.74)[0]
    n_chosen = len(selected_indices)
    print(f"Number of selected solutions: {n_chosen}")
    pf_merged_size, pf_merged_cr, pf_merged_val, pf_merged_test, pf_merged_n_ws  = np.zeros(n_chosen), np.zeros(n_chosen), np.zeros(n_chosen), np.zeros(n_chosen), np.zeros(n_chosen)
    for i, index in enumerate(selected_indices):
        compressed_model, centers, bin_indices, codebook, n_not_pruned = compression(pf_X[index], model, args)
        plot_histogram(centers, codebook, state="pre", args=args)
        
        compressed_model, centers, bin_indices, codebook = merge_bins(pf_X[index], compressed_model, val_loader, device=device)
        plot_histogram(centers, codebook, state="post", args=args)
        pf_merged_n_ws[i] = len(centers)
        torch.save({
                'state_dict': compressed_model.state_dict(),
                    'centers': centers,
                    'bin_indices': bin_indices,
                    'codebook': codebook
                    }, args.save_dir + f"/checkpoints/{args.merge_method}/merged_model_{index}index.pth.tar")
        compressed_model_size = size_of_model(compressed_model)
        pf_merged_size[i] = compressed_model_size
        cr = total_model_size / compressed_model_size
        pf_merged_cr[i] = cr
        pf_merged_val[i] = utils.evaluate(compressed_model, val_loader, eval=True, device=device) * 100
        pf_merged_test[i] = utils.evaluate(compressed_model, test_loader, eval=True, device=device) * 100
    

    res = pd.DataFrame({
        'K': pf_df['K'][selected_indices], 'alpha': pf_df['alpha'][selected_indices], 'beta': pf_df['beta'][selected_indices], 
        'n_pruned': pf_df['n_pruned'][selected_indices],
        'pre_merge_val_score': pf_val[selected_indices], 'post_merge_val_score': pf_merged_val,
        'pre_merge_test_score': pf_df['test_score'][selected_indices], 'post_merge_test_score': pf_merged_test,
        'pre_merge_size': pf_df['size'][selected_indices], 'post_merge_size': pf_merged_size,
        'pre_merge_cr': pf_df['CR'][selected_indices], 'post_merge_cr': pf_merged_cr,
        'pre_merge_n_ws': pf_df['n_weight_sharing'][selected_indices], 'post_merge_n_ws': pf_merged_n_ws,
    })

    res.to_csv(args.save_dir + f"/checkpoints/{args.merge_method}/pf_merged_summary.csv", index=False)

        




import os
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
from compression.quantization import get_compression, get_model_params, merge_bins, compression, uniform_quantization
import utils
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use(['science', 'no-latex'])

float32_bits = 32

args = arg_parser.parse_args()
# args.dataset = 'cifar10'
# args.seed = 6
args.model_path = f"./checkpoints/{args.dataset}/resnet18/0model_SA_best.pth.tar"
# args.save_dir = f"./logs/cifar100/resnet18/seed{args.seed}/"
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.save_dir+'/history/data', exist_ok=True)
os.makedirs(args.save_dir+'/history/figures', exist_ok=True)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(
    model,
    train_loader_full,
    val_loader,
    test_loader,
    marked_loader,
    ) = utils.setup_model_dataset(args)
model.cuda()
# checkpoint = torch.load(args.model_path, map_location=device)
# if "state_dict" in checkpoint.keys():
#     checkpoint = checkpoint["state_dict"]
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
    

def prune_and_ub_initialization(NP=100, problem=None):
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

if __name__ == "__main__":
    params = get_model_params(model)
    total_params = len(params)
    pretrained_acc = utils.evaluate(model, train_loader_full)
    print(pretrained_acc)
    total_model_size = size_of_model(model)
    print(total_params, total_model_size)

    if args.merge:
        print("merge enabled")

    problem = Problem(n_var=1, n_obj=2, n_constr=0, xl=np.array([2]), xu=np.array([1024]))
    # Define the algorithm
    NP = 100
    algorithm = NSGA2(
        pop_size=NP,
        crossover=SBX(prob=0.9, prob_var=0.5, eta=15),
        mutation=PM(prob=0.9, eta=20),
        eliminate_duplicates=True
    )

    algorithm.setup(problem, termination=NoTermination())

    # fix the random seed manually
    # np.random.seed(1)
    hv_tracker = []
    # Use the ask-and-tell interface
    n_steps = args.steps
    pbar = tqdm(range(0, n_steps), unit="step")
    for n_gen in pbar:
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()

        # if n_gen == 0:
        #     pop = pop.set("X", prune_and_ub_initialization(NP, lb_alpha, ub_beta, problem=problem))
        X = pop.get("X")
        X[:, 0] = X[:, 0].astype(int)
        pop.set("X", X)

        # implement your evluation
        f1, f2, c1, c2, c3 = np.zeros(NP), np.zeros(NP), np.zeros(NP), np.zeros(NP), np.zeros(NP)
        for i in range(NP):
            compressed_model, centers, bin_indices, codebook, n_not_pruned = compression(X[i], model, args)
            if args.merge:
                compressed_model, centers, bin_indices = merge_bins(X[i], model, train_loader_full)
            n_ws = len(centers)
            c1[i] = total_params - n_not_pruned
            c2[i] = n_ws
            f1[i] = 1 - utils.evaluate(compressed_model, train_loader_full)
            compressed_model_size = size_of_model(compressed_model)
            c3[i] = compressed_model_size
            cr = total_model_size / compressed_model_size
            f2[i] = -cr

            print(f"X: {X[i]}, score: {1-f1[i]}, CR: {f2[i]*-1}, n_prune: {c1[i]}")

        F = np.column_stack([f1, f2])
        static = StaticProblem(problem, F=F)
        Evaluator().eval(static, pop)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)

        res = algorithm.result()
        
        pf_F = res.F.copy()
        ind = HV(pf=pf_F)
        hv_val = ind(res.F)
        hv_tracker.append(hv_val)
        od = OrderedDict({'#_pf': len(res.F), 'HV': hv_val, 'ideal': res.F.min(axis=0), 'nadir': res.F.max(axis=0)})
        pbar.set_postfix(od)
        
        plot_pf(n_gen+1, pf_F, args)

        df = pd.DataFrame({"X_{0}": X[:, 0], "X_{1}": X[:, 1], "X_{2}": X[:, 2], "F_{0}": F[:, 0], "F_{1}": F[:, 1], "n_pruned": c1, "n_ws": c2, "size": c3})
        df.to_csv(args.save_dir + f'/history/data/step_{n_gen+1}.csv')

    res = algorithm.result()
    X = res.X
    F = res.F
    n_pf = len(res.X)
    test_F = np.zeros(n_pf)

    all_n_pruned = np.zeros(n_pf)
    all_n_ws = np.zeros(n_pf)
    for i in range(n_pf):
        compressed_model, centers, bin_indices, codebook, n_not_pruned = compression(X[i], model, args)
        if args.merge:
            compressed_model, centers, bin_indices = merge_bins(X[i], model, train_loader_full)
        n_ws = len(centers)
        test_F[i] = utils.evaluate(compressed_model, test_loader, eval=True)
        all_n_pruned[i] = total_params - n_not_pruned
        all_n_ws[i] = n_ws
    
    df = pd.DataFrame({"step": range(1, n_steps+1), "hv": hv_tracker})
    df.to_csv(args.save_dir + "/hv_tracker.csv", index=False)
    df = pd.DataFrame({"K": res.X[:, 0], "alpha": res.X[:, 1], "beta": res.X[:, 2], "train_score": 100*(1-res.F[:, 0]), "CR": -1*res.F[:, 1], "test_score": test_F * 100, "n_pruned": all_n_pruned, "n_ws": all_n_ws})
    df.to_csv(args.save_dir + "/pf_summary.csv", index=False)
    # Save the results
    torch.save({"pf_X": res.X, "pf_F": res.F, "hv_tracker": hv_tracker, "pf_test": test_F}, args.save_dir + "/results.pth.tar")
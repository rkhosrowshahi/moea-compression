import copy
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Tuple
from tqdm import tqdm

from compression.huffman_encoding import HuffmanEncode
from compression.pruning import range_prune
from utils import evaluate

# suppress Kmeans warning of memory leak in Windows
os.environ['OMP_NUM_THREADS'] = "1"

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import numpy as np
from scipy.sparse import csr_matrix, csr_array
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)


# Quantization base class inspired on torch.nn.utils.BasePruningMethod
class BaseQuantizationMethod(ABC):
    _tensor_name: str
    _shape: Tuple

    def __init__(self):
        pass

    def __call__(self, module, inputs):
        r"""Looks up the weights (stored in ``module[name + '_indices']``)
        from indices (stored in ``module[name + '_centers']``)
        and stores the result into ``module[name]`` by using
        :meth:`lookup_weights`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.lookup_weights(module))

    def lookup_weights(self, module):
        assert self._tensor_name is not None, "Module {} has to be quantized".format(
            module
        )  # this gets set in apply()
        indices = getattr(module, self._tensor_name + '_indices')
        centers = getattr(module, self._tensor_name + '_centers')
        # print(indices.shape, centers.shape)
        weights = F.embedding(indices, centers).squeeze()
        # print(weights.shape)yyyy
        ## debugging
        # weights.register_hook(print)
        if prune.is_pruned(module):
            mask = getattr(module, self._tensor_name + '_mask')
            mat = mask.detach().flatten().to(torch.float32)
            # print(mat[torch.argwhere(mat)].shape, weights.view(-1, 1).shape)
            # print(mat[torch.argwhere(mat)].type(), weights.view(-1, 1).type())
            mat[torch.argwhere(mat)] = weights.view(-1, 1)
        else:
            mat = weights
        return mat.view(self._shape)

    @abstractmethod
    def initialize_clusters(self, mat, n_points):
        pass

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        param = getattr(module, name).detach()
        # get device on which the parameter is, then move to cpu
        device = param.device
        shape = param.shape
        # flatten weights to accommodate conv and fc layers
        mat = param.cpu().view(-1, 1)

        # assume it is a sparse matrix, avoid to encode zeros since they are handled by pruning reparameterization
        mat = csr_matrix(mat)
        mat = mat.data
        if mat.shape[0] < 2 ** bits:
            bits = int(np.log2(mat.shape[0]))
            log.warning("Number of elements in weight matrix ({}) is less than number of clusters ({:d}). \
                        using {:d} bits for quantization."
                        .format(mat.shape[0], 2 ** bits, bits))
        # space = cls(*args, **kwargs).initialize_clusters(mat, 2 ** bits)

        # # could do more than one initialization for better results
        # kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
        #                 algorithm="lloyd")
        # kmeans.fit(mat.reshape(-1, 1))

        method = cls(*args, **kwargs)
        # Have the quantization method remember what tensor it's been applied to and weights shape
        method._tensor_name = name
        method._shape = shape

        # centers, indices = kmeans.cluster_centers_, kmeans.labels_
        centers = torch.nn.Parameter(torch.from_numpy(centers).float().to(device))
        indices = torch.from_numpy(indices).to(device)
        # If no reparameterization was done before (pruning), delete parameter
        if name in module._parameters:
            del module._parameters[name]
        # reparametrize by saving centroids and indices to `module[name + '_centers']`
        # and `module[name + '_indices']`...
        module.register_parameter(name + "_centers", centers)
        module.register_buffer(name + "_indices", indices)
        # ... and the new quantized tensor to `module[name]`
        setattr(module, name, method.lookup_weights(module))
        # associate the quantization method to the module via a hook to
        # compute the function before every forward() (compile by run)
        module.register_forward_pre_hook(method)
        # print("Compression rate for layer %s: %.1f" % compression_rate(module,name,bits))

    def remove(self, module):
        r"""Removes the quantization reparameterization from a module. The pruned
        parameter named ``name`` remains permanently quantized, and the parameter
        named  and ``name+'_centers'`` is removed from the parameter list. Similarly,
        the buffer named ``name+'_indices'`` is removed from the buffers.
        """
        # before removing quantization from a tensor, it has to have been applied
        assert (
                self._tensor_name is not None
        ), "Module {} has to be quantized\
                    before quantization can be removed".format(
            module
        )  # this gets set in apply()

        # to update module[name] to latest trained weights
        weight = self.lookup_weights(module)  # masked weights

        # delete and reset
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        del module._parameters[self._tensor_name + "_centers"]
        del module._buffers[self._tensor_name + "_indices"]
        module.register_parameter(self._tensor_name, torch.nn.Parameter(weight.data))


class UniformQuantizationMethod(BaseQuantizationMethod):
    def initialize_clusters(self, mat, n_points):
        return None

    @classmethod
    def apply(cls, module, name, centers, indices, *args, **kwargs):
        
        param = getattr(module, name).detach()
        device = param.device
        shape = param.shape
        # flatten weights to accommodate conv and fc layers
        mat = param.cpu().view(-1, 1)
        # assume it is a sparse matrix, avoid to encode zeros since they are handled by pruning reparameterization
        mat = csr_matrix(mat)
        mat = mat.data
        # return super(UniformQuantizationMethod, cls).apply(module, name, bits)
        method = cls(*args, **kwargs)
        # Have the quantization method remember what tensor it's been applied to and weights shape
        method._tensor_name = name
        method._shape = shape

        if name + "_centers" in module._parameters:
            method.remove(module)

        centers = torch.nn.Parameter(torch.from_numpy(centers).float().to(device))
        indices = torch.from_numpy(indices).to(device)
        # If no reparameterization was done before (pruning), delete parameter
        if name in module._parameters:
            del module._parameters[name]
        # reparametrize by saving centroids and indices to `module[name + '_centers']`
        # and `module[name + '_indices']`...
        module.register_parameter(name + "_centers", centers)
        module.register_buffer(name + "_indices", indices)
        # ... and the new quantized tensor to `module[name]`
        setattr(module, name, method.lookup_weights(module))
        # associate the quantization method to the module via a hook to
        # compute the function before every forward() (compile by run)
        module.register_forward_pre_hook(method)
        # print("Compression rate for layer %s: %.1f" % compression_rate(module,name,bits))

class LinearQuantizationMethod(BaseQuantizationMethod):
    def initialize_clusters(self, mat, n_points):
        min_ = mat.min()
        max_ = mat.max()
        space = np.linspace(min_, max_, num=n_points)
        return space

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        return super(LinearQuantizationMethod, cls).apply(module, name, bits)


class ForgyQuantizationMethod(BaseQuantizationMethod):
    def initialize_clusters(self, mat, n_points):
        samples = np.random.choice(mat, size=n_points, replace=False)
        return samples

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        return super(ForgyQuantizationMethod, cls).apply(module, name, bits)


class DensityQuantizationMethod(BaseQuantizationMethod):
    def initialize_clusters(self, mat, n_points):
        x, cdf_counts = np.unique(mat, return_counts=True)
        y = np.cumsum(cdf_counts) / np.sum(cdf_counts)

        eps = 1e-2

        space_y = np.linspace(y.min() + eps, y.max() - eps, n_points)

        idxs = []
        # TODO find numpy operator to eliminate for
        for i in space_y:
            idx = np.argwhere(np.diff(np.sign(y - i)))[0]
            idxs.append(idx)
        idxs = np.stack(idxs)
        return x[idxs]

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        return super(DensityQuantizationMethod, cls).apply(module, name, bits)


def uniform_quantization(module, name, centers, indices):
    UniformQuantizationMethod.apply(module, name, centers, indices)
    return module

def linear_quantization(module, name, bits):
    LinearQuantizationMethod.apply(module, name, bits)
    return module


def forgy_quantization(module, name, bits):
    ForgyQuantizationMethod.apply(module, name, bits)
    return module


def density_quantization(module, name, bits):
    DensityQuantizationMethod.apply(module, name, bits)
    return module


def is_quantized(module):
    for _, submodule in module.named_modules():
        for _, hook in submodule._forward_pre_hooks.items():
            if isinstance(hook, BaseQuantizationMethod):
                return True
    return False


def get_compression(module, name, idx_bits, huffman_encoding=False):
    # bits encoding weights
    float32_bits = 32

    all_weights = getattr(module, name).numel()
    n_weights = all_weights
    p_idx, q_idx, idx_size = 0, 0, 0

    if prune.is_pruned(module):
        attr = f"{name}_mask"
        mask = csr_array(getattr(module, attr).cpu().view(-1))
        n_weights = mask.getnnz()
        if huffman_encoding and n_weights > 0:
            try:
                # use index difference of csr matrix
                idx_diff = np.diff(mask.indices, prepend=mask.indices[0].astype(np.int8))
                # store overhead of adding placeholder zeros, then consider only indices below 2**idx_bits
                overhead = sum(map(lambda x: x // 2 ** idx_bits, idx_diff[idx_diff > 2 ** idx_bits]))
                idx_diff = idx_diff[idx_diff < 2 ** idx_bits]
                p_idx, avg_bits = HuffmanEncode.encode(idx_diff, bits=idx_bits)
                p_idx += overhead
                log.info(f" before Huffman coding: {n_weights*idx_bits:.0f} | after: {p_idx + overhead} | overhead: {overhead:.0f} | average bits: {avg_bits:.0f}")
            except:
                p_idx = n_weights * idx_bits
        else:
            p_idx = n_weights * idx_bits
    if is_quantized(module):
        attr = f"{name}_centers"
        n_weights = getattr(module, attr).numel()
        attr = f"{name}_indices"
        idx = getattr(module, attr).view(-1)
        # print(idx, sum(idx))
        weight_bits = np.ceil(math.log2(n_weights))
        q_idx = idx.numel() * weight_bits
        if huffman_encoding and len(idx) > 0:
            # print(weight_bits)
            # use index difference of csr matrix
            q_idx, _ = HuffmanEncode.encode(idx.detach().cpu().numpy(), bits=weight_bits)
        else:
            q_idx = len(idx) * weight_bits
    # Note: compression formula in paper does not include the mask
    return all_weights * float32_bits, n_weights * float32_bits + p_idx + q_idx


def compression_stats(model, name="weight", idx_bits=5, huffman_encoding=False):
    log.info(f"Compression stats of `{model.__class__.__name__}` - `{name}`:")
    compression_dict = {
        n: get_compression(m, name, idx_bits=idx_bits, huffman_encoding=huffman_encoding) for
        n, m in model.named_modules() if
        getattr(m, name, None) is not None}

    for name, (n, d) in compression_dict.items():
        cr = n / d
        log.info(f"  Layer {name}: compression rate {1 / cr:.2%} ({cr:.1f}X) ")
    n, d = zip(*compression_dict.values())
    total_params = sum(n)
    total_d = sum(d)
    cr = total_params / total_d
    log.info(f"Total compression rate: {1 / cr:.2%} ({cr:.1f}X) ")
    return cr


def get_model_params(model):
    param_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            for n in ["weight"]:
                param = getattr(layer, n)
                if param is not None:
                    # print(getattr(layer, n + '_mask'))
                    param_list.append(param.detach().cpu().numpy())
    param = np.concatenate([p.flatten() for p in param_list])
    return param

def set_model_params(model, centers, bin_indices):
    # compressed_model = copy.deepcopy(model)
    compressed_model = model

    start, end = 0, 0
    for name, layer in compressed_model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            for n in ["weight"]:
                param = getattr(layer, n)
                if param is not None:
                    param = param.detach()
                    device = param.device
                    shape = param.shape
                    mask = np.arange(shape.numel())
                    if prune.is_pruned(layer):
                        mask = getattr(layer, "weight" + '_mask').detach().flatten().cpu().numpy()
                    m_centers = centers.reshape(-1, 1)
                    m_indices = bin_indices[start: start+shape.numel()][mask]
                    uniform_quantization(layer, "weight", centers=m_centers, indices=m_indices)
                    start += shape.numel()

    return compressed_model

def get_center_and_indices(pre_model_param, bins):
    K = len(bins)
    bin_indices = np.digitize(pre_model_param, bins) - 1
    # print(bin_indices.min(), bin_indices.max())
    centers = []
    counter = 0
    codebook = {}
    bins_2 = []
    for i in range(K):
        
        mask = np.where(bin_indices == i)[0]
        if len(mask) == 0:
            continue
        bins_2.append(pre_model_param[mask].min()) 
        centers.append(pre_model_param[mask].mean())
        bin_indices[mask] = counter
        codebook[counter] = mask

        counter+=1
    centers = np.array(centers)
    bins_2[-1] = pre_model_param[mask].max()
    return centers, bin_indices, bins_2


def uniform_binning(K, model):
    # Get the model parameters
    pre_model_param = get_model_params(model)
    # pre_model_param = np.delete(pre_model_param, np.where(pre_model_param == 0))
    
    min_val = pre_model_param.min()
    max_val = pre_model_param.max()
    bins = np.linspace(min_val, max_val, K)
    bin_indices = np.digitize(pre_model_param, bins) - 1
    
    centers = []
    counter = 0
    codebook = {}
    for i in range(K):
        mask = np.where(bin_indices == i)[0]
        if len(mask) == 0:
            continue
        centers.append(pre_model_param[mask].mean())
        bin_indices[mask] = counter
        codebook[counter] = mask

        counter+=1
    centers = np.array(centers)
    set_model_params(model, centers, bin_indices)
                        
    
    return model, centers, bin_indices, codebook

def merge_bins_center_to_end(X, model, data_loader, device="cuda"):
    K, a, b = X
    K = int(K)
    
    pre_model_param = get_model_params(model)
    min_val = pre_model_param.min()
    max_val = pre_model_param.max()
    bins_curr = np.linspace(min_val, max_val, K)
    # bins_auto = np.histogram_bin_edges(pre_model_param, bins='auto')
    centers_curr, bin_indices_curr, bins_curr = get_center_and_indices(pre_model_param, bins_curr)
    model_curr = set_model_params((model), centers=centers_curr, bin_indices=bin_indices_curr)
    score_curr = evaluate(model_curr, data_loader, device=device)

    n_bins = len(centers_curr)
    pointer_left = (n_bins // 2)
    pointer_right = (n_bins // 2) + 1
    
    while pointer_right < n_bins - 1 and pointer_left > 1:
        n_bins = len(bins_curr)
        print(f"left p: {pointer_left}, right p: {pointer_right}, total: {len(bins_curr)}")
        bins_left = bins_curr.copy()
        bins_right = bins_curr.copy()
        # print(bins_left.shape, bins_right.shape)
        bins_left = np.delete(bins_left, pointer_left)
        # print(bins_left.shape)
        centers_left, bin_indices_left, bins_left2 = get_center_and_indices(pre_model_param, bins=bins_left)
        # print(len(centers_left), len(bin_indices_left), len(bins_left), len(bins_left2))
        model_left = set_model_params((model), centers=centers_left, bin_indices=bin_indices_left)
        score_left = evaluate(model_left, data_loader, device=device)

        bins_right = np.delete(bins_right, pointer_right)
        centers_right, bin_indices_right, bins_right2 = get_center_and_indices(pre_model_param, bins=bins_right)
        model_right = set_model_params((model), centers=centers_right, bin_indices=bin_indices_right)
        score_right = evaluate(model_right, data_loader, device=device)

        if score_left >= score_right and score_left >= score_curr:
            score_curr = score_left
            bins_curr = bins_left.copy()
            bin_indices_curr = bin_indices_left.copy()
            centers_curr = centers_left.copy()
            pointer_right -= 1
            pointer_left -= 1
            
        elif score_right >= score_left and score_right >= score_curr:
            score_curr = score_right
            bins_curr = bins_right.copy()
            bin_indices_curr = bin_indices_right.copy()
            centers_curr = centers_right.copy()

            pointer_right -= 1
            pointer_left -= 1
        else:
            pointer_right += 1
            pointer_left -= 1
    model = set_model_params((model), centers=centers_curr, bin_indices=bin_indices_curr)


    codebook = {}
    counter = 0
    for i in range(K):
        
        mask = np.where(bin_indices_curr == i)[0]
        if len(mask) == 0:
            continue

        codebook[counter] = mask
        counter += 1
    
    return model, centers_curr, bin_indices_curr, codebook



def merge_bins_left_to_right(X, model, data_loader, device="cuda"):
    K, a, b = X
    K = int(K)
    
    pre_model_param = get_model_params(model)
    min_val = pre_model_param.min()
    max_val = pre_model_param.max()
    bins_curr = np.linspace(min_val, max_val, K)
    # bins_auto = np.histogram_bin_edges(pre_model_param, bins='auto')
    centers_curr, bin_indices_curr, bins_curr = get_center_and_indices(pre_model_param, bins_curr)
    model_curr = set_model_params((model), centers=centers_curr, bin_indices=bin_indices_curr)
    score_curr = evaluate(model_curr, data_loader, device=device)

    n_bins = len(centers_curr)
    pointer = 2
    # pointer_right = (n_bins // 2) + 1
    
    iterator = tqdm(range(2, n_bins - 1), desc="Merging bins left to right")

    for pointer in iterator:
        # print(f"p: {pointer}, total: {len(bins_curr)}")
        iterator.set_description(f"Total bins: {len(bins_curr)}")
        
        bins_left = bins_curr.copy()
        bins_right = bins_curr.copy()

        bins_left = np.delete(bins_left, pointer-1)
        centers_left, bin_indices_left, _ = get_center_and_indices(pre_model_param, bins=bins_left)
        # print(len(centers_left), len(bin_indices_left), len(bins_left), len(bins_left2))
        model_left = set_model_params((model), centers=centers_left, bin_indices=bin_indices_left)
        score_left = evaluate(model_left, data_loader, device=device)

        bins_right = np.delete(bins_right, pointer)
        centers_right, bin_indices_right, _ = get_center_and_indices(pre_model_param, bins=bins_right)
        model_right = set_model_params((model), centers=centers_right, bin_indices=bin_indices_right)
        score_right = evaluate(model_right, data_loader, device=device)

        if score_left >= score_right and score_left >= score_curr:
            score_curr = score_left
            bins_curr = bins_left.copy()
            bin_indices_curr = bin_indices_left.copy()
            centers_curr = centers_left.copy()
            pointer -= 1
            
        elif score_right >= score_left and score_right >= score_curr:
            score_curr = score_right
            bins_curr = bins_right.copy()
            bin_indices_curr = bin_indices_right.copy()
            centers_curr = centers_right.copy()
            pointer -= 1
            # pointer += 1
        # else:
        #     pointer += 1
        if pointer >= len(bins_curr) - 2:
            break
        
    model = set_model_params((model), centers=centers_curr, bin_indices=bin_indices_curr)
    codebook = {}
    counter = 0
    for i in range(K):
        
        mask = np.where(bin_indices_curr == i)[0]
        if len(mask) == 0:
            continue

        codebook[counter] = mask
        counter += 1
    return model, centers_curr, bin_indices_curr, codebook



def compression(X, model, args):

    if len(X) == 2:
        K, a = X
        if a < 0:
            b = -a
        elif a > 0:
            b = a
            a = -a
    if len(X) == 3:
        K, a, b = X
    K = int(K)

    compressed_model = copy.deepcopy(model)
    n_not_pruned = len(get_model_params(model))
    # Apply weight sharing based on the codebook
    compressed_model, centers, bin_indices, codebook = uniform_binning(K, compressed_model)
    
    return compressed_model, centers, bin_indices, codebook, n_not_pruned
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

"""
from safe_gpu import safe_gpu


def use_single_gpu(gpus_qty: int) -> safe_gpu.GPUOwner:
    assert gpus_qty < 2, "Multi-GPU still not available."
    gpu_owner = safe_gpu.GPUOwner(nb_gpus=gpus_qty)
    return gpu_owner
"""

class GPUOwner:
    def __init__(self, nb_gpus: int):
        self.nb_gpus = nb_gpus

def use_single_gpu(gpus_qty: int) -> GPUOwner:
    if gpus_qty < 2:
        # Set the environment variable to use only the first GPU
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("Using a single GPU.")
        return GPUOwner(nb_gpus=1)
    else:
        raise ValueError("Multi-GPU still not available.")

def use_single_gpu(gpus_qty: int) -> dict:
    if gpus_qty < 2:
        # Set the environment variable to use only the first GPU
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("Using a single GPU.")
        return {'nb_gpus': 1}
    else:
        raise ValueError("Multi-GPU still not available.")


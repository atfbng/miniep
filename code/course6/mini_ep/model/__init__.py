"""本课使用的公共接口。"""
from mini_ep.model.moe import SingleGPUMoE
from mini_ep.model.moe import distribute_single_gpu_moe

__all__ = ['SingleGPUMoE', 'distribute_single_gpu_moe']

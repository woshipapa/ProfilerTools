import torch
import sys
import torch.distributed as dist
import time
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
import hashlib
import torch.nn as nn
from .logging import get_logger

def print_model_params(model):
    print("Model Parameters:")
    print("=" * 50)
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor):
            print(f"Layer: {name}")
            print(f"Shape: {param.shape}")
            print(param.data)  # 仅打印数值，不计算梯度
            print("-" * 50)

def tensor_md5(tensor: torch.Tensor) -> str:
    tensor = tensor.to(torch.float64)
    # 确保 Tensor 在 CPU 上，并转换为 numpy 数组
    tensor_np = tensor.detach().cpu().numpy()
    # 将 numpy 数组转换为 bytes
    tensor_bytes = tensor_np.tobytes()
    # 计算 MD5
    md5_hash = hashlib.md5(tensor_bytes).hexdigest()
    return md5_hash

class DebugLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


filename = "_output_14_backward_2.log"

def register_hooks(model, print_values=True, max_elements=20):
    hooks = []
    
    def print_shape_and_values(tensor, label, max_elements=20):
        """Helper function to print shape and values of a tensor"""
        rank = 0
        with open(str(rank) + filename, "a") as f:
            if isinstance(tensor, torch.Tensor):
                get_logger().info(f"[rank {rank}] {label} shape: {tensor.shape}")
                if print_values:
                    if tensor.numel() < max_elements:
                        max_elements = tensor.numel()
                    if tensor.flatten()[0].dtype != torch.bool:
                        # get_logger().info(f"[rank {rank}]  {label} first 20 values: {tensor.flatten()[:max_elements]}{'...' if tensor.numel() > max_elements else ''} ")
                        # x, _ = torch.topk(tensor.flatten(), max_elements)
                        # get_logger().info(f"[rank {rank}]  {label} max 20 values: {x} ")
                        # x, _ = torch.topk(-tensor.flatten(), max_elements)
                        # get_logger().info(f"[rank {rank}]  {label} min 20 values: {x} ")
                        tensor = tensor.float()
                        x = torch.norm(tensor)
                        get_logger().info(f'[rank {rank}] {label} norm Values: {x} ')
                        # get_logger().info(f'[rank {rank}] {label} md5 value: {tensor_md5(tensor)} ')
                        get_logger().info(f'[rank {rank}] {label} shape: {tensor.shape} ')
            else:
                get_logger().info(f"[rank {rank}]  {label} is not tensor, is {tensor} ")

    def forward_hook_fn(module, input, output, name):
        """Hook 函数，打印输入和输出的 shape 和具体数值"""
        # print(f"Layer: {module.__class__.__name__}")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name}")
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                print_shape_and_values(weight, f"{name}_Weight")
            else:
                get_logger().info(f"[rank {rank}] Layer: {name} has no weight ")
        # 打印输入
        if isinstance(input, tuple):
            for idx, inp in enumerate(input):
                if isinstance(inp, tuple):  # Check for nested tuple
                    for sub_idx, sub_inp in enumerate(inp):
                        print_shape_and_values(sub_inp, f"Input {idx}-{sub_idx}")
                else:
                    print_shape_and_values(inp, f"{name}_Input {idx}")
        else:
            print_shape_and_values(input, "Input")

        # Print output shapes and values
        if isinstance(output, tuple):
            for i, tensor in enumerate(output):
                if isinstance(tensor, tuple):  # Check for nested tuple
                    for sub_idx, sub_tensor in enumerate(tensor):
                        print_shape_and_values(sub_tensor, f"{name}_Output {i}-{sub_idx}")
                else:
                    print_shape_and_values(tensor, f"{name}_Output {i}")

        else:
            print_shape_and_values(output, "Output")
        with open(str(rank) + filename, "a") as f:
            get_logger().info("-" * 100)
            get_logger().info(" ")
        

    def backward_hook_fn(module, grad_input, grad_output, name):
        """Hook 函数，打印反向传播时的梯度"""
        # print(f"Layer: {module.__class__.__name__} (backward)")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name} (backward)")
            # 打印输入梯度
            for idx, grad in enumerate(grad_input):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Input {idx}")

            # 打印输出梯度
            for idx, grad in enumerate(grad_output):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Output {idx}")
            
            # if hasattr(module, 'weight') and module.weight is not None:
            #     print_shape_and_values(module.weight.grad, f"{name}_weight_grad")
            # if hasattr(module, 'bias') and module.bias is not None:
            #     print_shape_and_values(module.bias.grad, f"{name}_bias_grad")
            
            get_logger().info("-" * 100)
            get_logger().info(" ")

    # 遍历模型的所有层并注册 hook
    # for layer in model.modules():
    #     if not isinstance(layer, torch.nn.Sequential) and not isinstance(layer, torch.nn.ModuleList):
    #         forward_hook = layer.register_forward_hook(forward_hook_fn)
    #         backward_hook = layer.register_backward_hook(backward_hook_fn)
    #         hooks.append(forward_hook)
    #         hooks.append(backward_hook)
    
    # if dist.is_available() and dist.is_initialized():
    #     rank = 0
    # else:
    #     rank = 0

    def watch_parameter(param_name, param):
        rank = 0
        def param_hook(grad):
            if grad is not None:
                with open(str(rank) + filename + "_grad", "a") as f:
                    get_logger().info(f"{param_name} grad norm: {grad.norm()} ")
            else:
                with open(str(rank) + filename + "_grad", "a") as f:
                    get_logger().info(f"{param_name} grad is None ")
        param.register_hook(param_hook)

    for name, module in model.named_modules():
        forward_hook = module.register_forward_hook(lambda m, i, o, name=name: forward_hook_fn(m, i, o, name))
        backward_hook = module.register_full_backward_hook(lambda m, i, o, name=name: backward_hook_fn(m, i, o, name))
        hooks.append(forward_hook)
        hooks.append(backward_hook)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            watch_parameter(name, param)
    
    
    return hooks  # 返回 hook 句柄列表，方便后续清理


import time,os


class MyTimer:
    def __init__(self, use_cuda=True, tag="timer", verbose=True, log_dir="./time_log"):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.tag = tag
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.log_dir = log_dir

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = os.path.join(self.log_dir, f"{self.tag}_rank{self.rank}.log")
        else:
            self.log_path = None

        # 多阶段时间字典
        self._stage_times = {}  # stage_name -> {"cpu_start": ..., "cuda_start": ..., "cuda_end": ...}
        self.records = []

    def start(self, stage_name):
        entry = {}
        
        entry["cpu_start"] = time.time()
        self.rank = dist.get_rank()
        print(f"Rank {dist.get_rank()} {stage_name} start at time {entry['cpu_start']}")
        if self.use_cuda:
            entry["cuda_start"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_end"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_start"].record()
        self._stage_times[stage_name] = entry

    def stop(self, stage_name):
        entry = self._stage_times.get(stage_name, {})
        cpu_end = time.time()
        cuda_elapsed = None

        if self.use_cuda and "cuda_end" in entry:
            entry["cuda_end"].record()
            torch.cuda.synchronize()
            cuda_elapsed = entry["cuda_start"].elapsed_time(entry["cuda_end"])  # ms

        cpu_elapsed = cpu_end - entry.get("cpu_start", cpu_end)

        self.records.append({
            "stage": stage_name,
            "rank": self.rank,
            "cpu_start": entry.get("cpu_start", None),
            "cpu_end": cpu_end,
            "cpu_duration": cpu_elapsed,
            "cuda_duration": cuda_elapsed
        })

        if self.verbose:
            print(f"[Rank {self.rank}] Stage {stage_name}: CPU {cpu_elapsed:.6f}s, CUDA {cuda_elapsed:.3f}ms")

    def dump(self):
        if self.log_path:
            with open(self.log_path, "a") as f:
                header = f"\n================================ DUMP START  ================================\n" 
                f.write(header)
                for r in self.records:
                    f.write(
                        f"[RANK {r['rank']}] {r['stage']}, start={r['cpu_start']:.6f}, "
                        f"end={r['cpu_end']:.6f}, cpu_dur={r['cpu_duration']:.6f}s, "
                        f"cuda_dur={r['cuda_duration']:.3f}ms\n"
                    )


global_timer = MyTimer()        
# ProfilerTools
used to profile cuda memory and time in distributed training

本项目包含两个核心性能分析组件：

- [🔍 ProfilerWrapper](#-profilerwrapper-transformer-性能分析工具)
- [⏱ global_timer](#-global_timer-分布式时间记录工具)
## 🔍 ProfilerWrapper: Transformer 性能分析工具

`ProfilerWrapper` 是专为 Transformer 结构的 PyTorch 深度学习模型开发的性能分析与统计工具。它可以自动追踪和汇总 Transformer 各层（如 attention/block/FFN）的 CUDA 计算耗时、显存使用、Flash Attention 执行情况等关键性能指标，帮助你精准发现瓶颈与优化方向。

支持单机/多卡分布式 (`torch.distributed`)，可输出丰富的统计表、可视化曲线，并支持与主流大模型（如 CogVideo、OpenSora、BERT/GPT/ViT）无缝集成。

---

## 主要特性

* 针对 **Transformer 每一层**，自动统计 CUDA/CPU 执行时间与显存消耗
* 支持单步/多轮 profiling，自动按层、按迭代输出曲线与表格
* 可区分统计 `spatial block` / `temporal block` / `cogvideo block` / `aten::_flash_attention_forward` 等常见子模块
* 支持采集和导出 Chrome Trace（便于火焰图、timeline 分析）
* 一键输出显存折线图、各层自定义统计表，支持 FlashAttention 单独跟踪
* 多参数灵活配置，适用于大模型训练/推理、效率对比、资源瓶颈定位等多种场景

---

## 应用场景

ProfilerWrapper 适合如下典型任务：

* Transformer/Attention 网络（如 BERT、GPT、ViT、Diffusion/Video Model）多层性能逐层分析
* 针对长序列、多层堆叠网络，统计每层的运行时间与显存变化，辅助寻找慢层/爆显存层
* FlashAttention/自定义 block 的耗时跟踪、分布式多卡多轮性能对比
* 优化论文实验、自动生成 profiler 汇总图表，支撑性能可视化展示

---

## 快速上手示例

```python
from profiler_wrapper import ProfilerWrapper

def forward_fn():
    # 你的 transformer 前向/反向逻辑
    pass

with ProfilerWrapper(
    is_st=False,
    enabled_record_memory_pickle=True,
    enabled_record_cuda_average_time=True,
    enable_record_cuda_mm=True,
    enable_print_summary=True
) as profiler:
    profiler.record_function('block', forward_fn)
    # 或直接把模型前向/反向放在 with 体内
    # ...
# 结束后自动输出统计结果和可视化曲线
```

---

## 参数说明

| 参数                                 | 作用                                  | 典型用途                    |
| ---------------------------------- | ----------------------------------- | ----------------------- |
| `is_st`                            | 文件名前缀（True=opensora，False=cogvideo） | 区分不同模型日志                |
| `activities`                       | 监控设备列表，默认 CPU+CUDA                  | 控制 profiler 活动类型        |
| `profile_memory`                   | 是否采集内存曲线                            | 显存泄漏/峰值分析               |
| `record_shapes`                    | 记录算子输入 shape                        | 溯源高耗时层                  |
| `log_dir`                          | 日志/汇总输出目录                           | profiler 汇总/表格          |
| `mem_dir`                          | 内存快照输出目录                            | pickle/memory 诊断        |
| `enabled_record_memory_pickle`     | 使用pytorch的snapshot功能保存显存变化pickle图                        | 完整的一段显存可视化分析                  |
| `enabled_record_cuda_average_time` | 支持自定义的从每层的profiler得到的kernel或者record的name中，自己做数据的处理得到一个时间                      | 统计每层具体kernel的时间，可以可视化 |
| `enable_record_cuda_mm`            | 采集每一层的allocated,reserved memory，最终绘图                          | 多层显存分析，主要用于判断每层激活值显存的增加                |
| `enable_print_summary`             | 保存/打印 profiler 表格                   | 查找慢算子                   |

---

## 输出文件

* `*_trace.json` —— Chrome Trace，支持 [chrome://tracing](chrome://tracing)
* `*_cuda_time_plot.png` —— 各层 CUDA 自耗时曲线（支持 FA 统计）
* `*_memory_usage_line_chart.png` —— 显存分配/保留曲线
* `*_memory_usage.txt` —— 逐轮显存分布文本
* `log_dir/*_rank_*.log` —— profiler 汇总表格

---

## 典型用法

```python
with ProfilerWrapper(
    is_st=True,  # or False
    enabled_record_cuda_average_time=True,
    enable_record_cuda_mm=True,
    enable_print_summary=True
) as prof:
    prof.record_function('my_forward', your_forward_fn)
    # 或者直接写在with体内：your_forward_fn()
```

---

## 常见问题与建议

* **适合多层 Transformer、深堆叠网络逐层分析。建议与 `torchrun`/`torch.distributed.launch` 配合，多 rank 自动区分输出**
* 内存/时序统计可分阶段开启，建议分析瓶颈时按需打开
* 依赖 `matplotlib`、`numpy`，需提前安装
* 各项输出文件均可在实验报告/论文可视化中直接引用

---




























## ⏱ `global_timer` 使用说明

`MyTimer` 是一个分布式多进程环境下的通用时间记录工具，用于 **精确测量并记录各个阶段的运行时间（支持 CPU 和 CUDA）**，并将每个 rank 的日志分别写入文件，后续可以通过正则匹配与 DataFrame 分析构建多进程时间轴（timeline）可视化。

---

## ✅ 功能概述

- ✅ 支持 CPU + CUDA 计时（`time.time()` + `torch.cuda.Event`）
- ✅ 支持多个命名阶段（`forward`、`backward` 等）
- ✅ 支持多进程 rank 自动识别和独立输出日志
- ✅ 自动创建日志目录和文件，避免冲突
- ✅ 日志可用于 timeline 可视化、straggler 分析

---

## 🧱 使用方式

```python
from utils import global_timer

# 开始记录某阶段
global_timer.start("forward")

# 模块逻辑
...

# 停止记录该阶段
global_timer.stop("forward")

# 最后导出所有记录
global_timer.dump()
```

## 📝 日志输出格式
每个 rank 会生成一个 .log 文件，内容如下：
```sql
[RANK 0] forward, start=1716182684.282194, end=1716182685.123003, cpu_dur=0.840809s, cuda_dur=830.512ms
```
## 📁 输出文件结构示例
```lua
time_log/
├── timer_rank0.log
├── timer_rank1.log
└── ...
```

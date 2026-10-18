# 多 LoRA 草稿模型投机推理预实验

> **研究进度更新（2026-10-18）**：仓库原有根目录文件和 2026-10-17 发布的 `seed42-snapshot.tar.gz` 是第一版的固定快照。新增的 [当前研究协议](RESEARCH_PROGRAM_CURRENT.md)、[第二版五任务代码和独立协议](experiments/v2-routing/program.md)、[第二版结果交接](reports/v2-routing/handoff.md) 及 [第三版拒绝预测／预算路由试验](experiments/v3-async-budget/RESULTS.md) 是后续独立工件；不会改写原快照的 P0 哈希。第二版因在线净时延 P3 未通过而停止；第三版仅发现离线拒绝预测信号，分支和预算专家策略未证明加速，也没有执行真实 WAN 或 RL 训练。第三版原始逐请求诊断与有界迭代记录同仓保存。

Qwen3-0.6B / Qwen3-4B 草稿主干、领域 LoRA 与混合领域 LoRA 的 seed42 预实验。目标模型为 Qwen3-8B。本仓库是可追溯的实验快照，包含训练、教师数据、选模、专家接受矩阵、路由和真实 greedy 投机解码代码。

**快照状态：2026-10-17 06:33 UTC。10/10 组训练和选模完成；L8 专家矩阵评测进行中，真实接受路由与在线对比尚未完成。不能把此快照视为两个预实验的最终结论。** 原 AutoDL 和 4090 服务未因归档而停止。快照之后的运行进度不自动等于本仓库中的状态。

## 获取代码、数据和实际训练权重

代码与文档可以直接浏览；[Releases](../../releases) 保存完整快照 `seed42-snapshot.tar.gz`、SHA256 清单和实验说明。解压后顶层为 `snapshot/`。

- `adapters/{0.6B,4B}/{math,code,general,mix16,mix48}/seed42/overnight-20261017-seed42/`：**10 个选中 LoRA 的真实 safetensors 权重**、adapter_config、selected.json、训练完成记录。每组只保留选中的 epoch。
- `data/teacher/{math,code,general}/`：8100 条已冻结教师记录，包含 token ID、来源及校验信息；无需重新生成。
- `data/processed/`：训练、adapter_val、router_train、router_val 固定划分；最终测试划分只作冻结备份，仍保持锁定，不参与本次训练、选模或评测。
- `results/validation/`：逐样本、逐 epoch 的选模接受记录；`results/continuation-20261017/` 为导出时已生成的阶段结果。
- `results/smoke/report.json`、`manifests/p0_gate.json`：通过的正确性检查及源码绑定信息。
- `SNAPSHOT.json`：导出时间、所选 epoch、权重 SHA256、当时真实进程阶段及排除项。

不上传未选中 epoch、优化器/断点大文件、失败的历史实现、登录日志、SSH 密钥或下载缓存。保留会影响科学结论的限制说明。官方 Qwen 基础模型不重复上传；其固定版本和下载方法见 [environment/README.md](environment/README.md)。上传的模型权重指本实验实际训练出的 LoRA，使用时仍需对应主干。

## 环境

原运行环境：Linux、Python 3.11、Conda `spec`、PyTorch 2.8.0/CUDA 12.8、Transformers 4.57.3、PEFT 0.18.0、Datasets 4.4.1。精确实际版本以 `environment/requirements-spec.txt` 和 `manifests/environment/spec-conda-explicit.txt` 为准。

```bash
conda create -p /root/miniconda3/envs/spec python=3.11 -y
conda activate /root/miniconda3/envs/spec
python -m pip install -r environment/requirements-portable.txt
```

源代码为保留 P0 来源哈希而保持原样，运行根目录为 `/root/autodl-tmp/sd-lora-preexp`，Python 前缀为 `/root/miniconda3/envs/spec`。在**新机器**恢复时，应将快照内容放到该目录，并按配置下载基础模型。不同路径的迁移不能直接沿用旧 P0 gate，需重新检查。不要在已有活跃实验目录覆盖文件或重复启动监督器。

```bash
cd /root/autodl-tmp/sd-lora-preexp
/root/miniconda3/envs/spec/bin/python -m pytest -q scripts/continuation/test_evaluation_helpers.py
/root/miniconda3/envs/spec/bin/python smoke_test.py --help
```

9 项 CPU 编排检查不代表 GPU 性能结果，也不能替代新机器上的完整 token 一致性检查。恢复说明见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 实验问题与配置

1. **主干规模与专门化的收益—成本曲线**：0.6B/4B，未微调 BASE、单领域专家、MIX16、MIX48；比较接受长度、草稿成本、显存与最终确认 token 的速度。
2. **专家互补、真实路由与部署成本**：离线 Oracle、固定最佳专家、任务路由、接受收益路由，区分共享主干多 LoRA 与真正独立副本。Oracle 只能报告离线接受上限。

| 项目 | 冻结设置 |
|---|---|
| 目标模型 | Qwen3-8B，BF16，greedy，关闭 thinking，SDPA |
| 草稿主干 | Qwen3-0.6B、Qwen3-4B，BF16 |
| 每规模适配器 | math、code、general、mix16、mix48 |
| LoRA | q_proj/v_proj；r16/alpha32；mix48 为 r48/alpha96；dropout0.05 |
| 训练 | seed42，2 epoch，AdamW lr1e-4，microbatch1，累积16，max_seq2048 |
| 数据 | GSM8K、Magicoder Python、UltraChat；每领域2000训练/200选模/400路由训练/100路由验证 |
| 选模 | 每请求平均接受长度，再领域宏平均；epoch1/2择优，同分取较早 epoch |
| 接受窗口 | L8，公共前缀位置0/32/64/128/192，仅计足够长的有效窗口 |

教师生成是离线数据准备；在线实验会实际执行“小模型生成草稿 → 大模型批量验证 → 拒绝后修正”，并与 AR 完整 token/EOS/截断结果逐请求严格比较。

## 本次在线范围及限制

用户授权的当前阶段为 **每领域8条，共24条，L8，单次计时**，两规模七方法共336次 SD 对比。每规模只独立测量24条目标模型 AR，再供七方法按相同请求配对复用，不把复用引用算成七份独立基线。抽样固定 seed20261017；不根据输出筛选样本。组件 profiling 单独运行，不混入正常时延。

**原 `program.md` 未修改**，SHA256：`6e119661db905a609565363b6b75d4c5a3a8b4f7544940dc8d51d4b1774f4b43`。完整600请求在线、L4/L12扫描、更多计时重复、seed43/44、最终测试和真实云边 WAN/SLO 实验均延期。24条一次计时只能判断初步趋势。

历史部署检查中，BF16 合并完整副本与动态 LoRA 存在 token 输出差异，不能声称二者严格等价，也不能据此发布等价条件下的速度结论。当前无完整在线结果，不承诺多专家一定胜过混合 LoRA。

数据及基础模型遵循各自上游许可；本私有实验备份不对上游数据或模型重新授权。

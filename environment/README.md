# spec 环境与基础模型

原环境 Python 3.11.17，PyTorch 2.8.0 / CUDA 12.8，Transformers 4.57.3，PEFT 0.18.0。
requirements-spec.txt 是实际 pip freeze 原件；requirements-portable.txt 仅去掉不可跨机器使用的本地构建 file:// packaging 条目，packaging 由依赖解析安装。Conda 精确锁位于 manifests/environment/spec-conda-explicit.txt。环境重建尚未在全新机器验证，安装后必须运行环境和正确性检查。

```bash
conda create -p /root/miniconda3/envs/spec python=3.11 -y
conda activate /root/miniconda3/envs/spec
python -m pip install -r environment/requirements-portable.txt
python tools/download_base_models.py
```

基础模型使用上游 Qwen/Qwen3-0.6B、Qwen3-4B、Qwen3-8B：

| 模型 | revision |
|---|---|
| Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca |
| Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |

下载脚本使用 Hugging Face 官方接口，写入原配置目录；只在新机器恢复时运行，不能覆盖活跃实验的权重目录。基础权重未复制到本仓库，实际训练的10个LoRA已附在Release快照中。跨GPU、驱动、CUDA或算子后端时需重跑正确性门槛，不默认BF16输出逐token相同。

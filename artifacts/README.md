# 第二版选中权重与教师轨迹备份

`v2-selected-20261018.tar.gz` 是 AutoDL 实验结束前的固定归档（SHA256 `4f31871ae3496fe7f040dea2ec490279d03ab549f7a96bb9306e7e419653a06b`，97,205,786 字节）。包含 0.6B 的五任务专家、MIX16、MIX80 共 7 个**已选中** LoRA 的 `adapter_model.safetensors`／配置／`selected.json`，以及 2,120 条冻结教师轨迹、pilot manifest 和最终报告／正确性审计；共 2,145 个源文件。归档内 `V2_SELECTED_SHA256SUMS.json` 列出每文件哈希。

这里的模型权重是 LoRA 增量，**不含官方 Qwen3 基础模型**，也不含未选中 epoch、优化器或训练中间检查点。解压后顶层是 `experiments/v2-routing/`；`selected.json` 保留原 AutoDL 的绝对路径用于审计，迁到别的机器需要按新路径更新加载配置并重做环境与 token 正确性检查。第二版 P3 净时延门槛未通过，不要把这个权重备份理解为部署推荐。

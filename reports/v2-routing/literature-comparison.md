# 第二版预实验与 TaskSpec、MetaSD 的指标对照

整理日期：2026-10-18。跨论文数字仅用于定位研究问题，**不能当作同条件性能排名**。我们没有复现 TaskSpec 或 MetaSD 的原始代码、模型和数据；`META-UCB` 只是受 MetaSD 启发的 UCB 基线。

## 协议与主要结果

| 项目 | TaskSpec | MetaSD（黑盒 MetaSpS-UCB） | 我们第二版预实验 |
|---|---|---|---|
| 目标／草稿 | LLaMA-2-13B／LLaMA-68M；任务专门微调草稿 | Vicuna-7B／5 个任务专门的 Vicuna-68M 草稿 | Qwen3-8B／Qwen3-0.6B 共享主干，5 个领域 LoRA；另有同数据 MIX16、MIX80 |
| 选择方式 | prompt 任务分类后，按请求选草稿 | 利用验证反馈，UCB 在生成过程中选草稿 | DOMAIN、prompt-only ROUTE-TASK／ROUTE-ACC、META-UCB 风格及 COST-UCB 候选 |
| 任务 | 4 类：中文文本生成、数学、英译、化学问答 | 5 类：代码、德英翻译、摘要、问答、数学 | 同类 5 任务；问答为 NQ-Open 变体，来源与其主实验不完全一致 |
| 硬件与设置 | 单张 A100 40GB，batch 1、greedy，3 次独立评测；按任务调 draft length | 黑盒主表单张 A6000、greedy，通常最多 5 个 drafts | 单张 RTX 6000D、BF16、batch 1、greedy、L=8；在线每任务 8 条，单次计时 |
| 接受质量 | 报接受率和平均接受长度；对 vanilla SpecDec 的接受质量有提升 | 使用接受反馈选择草稿；主表重点是加速比 | 240 条 adapter_val 的 L8 矩阵：MIX80 平均连续接受 3.792，DOMAIN 3.784，事后 ORACLE 4.036 token／公共窗口；Oracle−MIX80 为 +0.245 [0.093, 0.387] |
| **实测速度** | 图 10 报相对直接 AR 的墙钟加速；四任务最高值约 1.47、2.64、1.76、1.81 倍，分别在不同 draft length 取得 | 表 3 报相对直接 AR 的加速；UCB 五任务分别为 2.300、1.587、1.971、1.711、2.280 倍 | 40 条在线请求：MIX80 为 **0.373 倍 AR**，META-UCB 风格为 **0.364 倍 AR**，COST-UCB 为 **0.356 倍 AR**；COST-UCB／MIX80 为 **0.955** [0.913, 0.990] |
| 云边／网络 | 未将真实云边链路计入这些主结果 | 未将真实云边链路计入这些主结果 | 未测真实 WAN；同卡顺序执行 |

TaskSpec 图 10 的标题以 AR 为速度基线；其正文同时叙述了相对 vanilla SpecDec 的改善。引用该文时必须明确对应图和基线，不能把两种比值混写。MetaSD 表 3 的速度基线明确为标准自回归 greedy。

## 我们自己的同条件对照

下表均为同一组 40 条 `router_val` 请求的真实在线结果，8 条／任务，L=8、一次计时；平均接受长度与上表的 240 条离线矩阵**不是同一个样本统计量**。

| 方法 | 在线平均每轮接受草稿 token | 速度／8B AR（五任务几何平均） | 相对 MIX80 速度 |
|---|---:|---:|---:|
| 原始 0.6B 草稿 BASE | 3.206 | 0.409 | — |
| 混合 LoRA MIX16 | 3.303 | 0.368 | — |
| 混合 LoRA MIX80 | 3.342 | 0.373 | 1.000 |
| 已知领域 DOMAIN | 3.300 | 0.356 | — |
| prompt-only ROUTE-ACC | 3.300 | 0.366 | — |
| MetaSD 风格 META-UCB | 3.285 | 0.364 | 0.974 [0.930, 1.015] |
| 我们的 COST-UCB 候选 | 3.231 | 0.356 | 0.955 [0.913, 0.990] |

所有 8 种 SD 方法在上述 40 条请求上与独立 AR8 的完整 token、EOS、截断一致，合计 320 次逐请求比对零分歧。40 条 AR 计时按同一请求配对复用，**不是 8 组独立 AR 测量**。这里的速度比小于 1 表示比直接生成慢。

## 可以支持和不能支持的结论

1. 五专家的事后 Oracle 优于等数据 MIX80，支持“有互补空间”；它事先知道目标输出，不能证明实际路由可兑现收益。prompt-only 路由在 120 条 router_val 上平均接受长度比 MIX80 低约 0.102 token。
2. COST-UCB 在这次在线预实验中慢于 MIX80 和 AR8，因此预注册 P3 净时延 gate 未通过；按协议没有启动 P4 扩量、多 seed 或最终测试。
3. TaskSpec 和 MetaSD **都报告实际速度**，不是只报接受率。它们的模型、草稿成本、任务、长度、硬件和计时重复与我们不同；不能把它们的加速比除以我们的 0.356 来声称算法差距。
4. [E²-SCI](https://openaccess.thecvf.com/content/CVPR2026/html/Li_E2-SCI_Elastic_Edge-Cloud_Speculative_Decoding_via_Credit_Inertia_CVPR_2026_paper.html) 是另一个边云系统参照，报告异步流水线与阈值策略在边云配置下的吞吐／时延；其数字和上述单卡专家路由表不共用基线或正确性口径。

## 可核查来源

- [TaskSpec 原文：实验配置、指标与图 10](https://arxiv.org/html/2505.08600)
- [MetaSD 原文：表 3、实验配置与算法](https://arxiv.org/html/2604.05417)
- [我们的质量矩阵](quality_summary.json)
- [我们的静态在线报告](online_summary.json)
- [我们的反馈在线报告](feedback_summary.json)
- [我们的终止报告与门槛结果](final_report.json)
- [我们的逐请求正确性审计](final_audit.json)

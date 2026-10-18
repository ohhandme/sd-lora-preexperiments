# 多专家草稿与路由研究协议

> **2026-10-17 更新：**下文 1.1–1.17 是已冻结的第一版协议，原文留作审计，不再按其中“GPU 尚未执行”等旧状态句判断当前进度。第一版 seed42 的有限清单已经完成，实测总结和第二版新协议见末尾第 2 节。原版原文已归档为 `work/program-v1.0-frozen-20261017.md`，SHA256 `6e119661db905a609565363b6b75d4c5a3a8b4f7544940dc8d51d4b1774f4b43`。第二版必须使用独立数据/结果目录及新的协议哈希；第一版 P0 gate、源码、模型权重与结果不因文档更新而追认变更。

# 第一研究点预实验：常驻多 LoRA 草稿专家的质量、计算与显存权衡

协议版本：v1.0；制定日期：2026-10-15；状态：**实验方案已制定，GPU 实验尚未执行，本文不包含实测结论**。

本文是研究执行协议，不是 Karpathy 原版 nanochat 预训练脚本。使用 autoresearch 的固定评测、单因素迭代、完整日志与保留/淘汰机制；不照搬其 5 分钟训练或 val_bpb 指标。本轮交付仅为制定协议和安装 skill，不启动租卡、模型下载、训练或无限实验循环。

## 1.1 MOTIVATION、问题与可证伪假设

目标：判断在有限边缘计算和显存预算下，共享较大草稿主干并配置领域 LoRA，能否比通用草稿或多个较小完整专家更有效地生成被固定目标模型接受的候选。

动机分为三层：

1. 领域专门化可能提高草稿与目标的对齐，但必须超过使用相同数据训练的混合领域通用草稿，不能只超过未微调模型。
2. 多 LoRA 减少同等规模专家之间的主干重复存储，可能允许采用更大的共享主干；但 4B 主干并不比三个 0.6B 完整专家更省显存。
3. 更大草稿可能提高接受长度，也增加生成成本；接受率提升必须转换为最终确认输出的实际加速。

预先登记假设，不写成已有发现：

| 编号 | 假设 | 被否证的情况 |
|---|---|---|
| H1 | 相同主干下，领域专家具有可利用的接受质量互补 | 某一固定专家或混合领域单 LoRA 基本占优，oracle 收益很小 |
| H2 | 4B 共享多 LoRA 的接受收益能够覆盖相对 0.6B 的额外计算 | 仅接受率提高，确认输出速度未改善 |
| H3 | 共享主干部署保留专门化收益，并减少相对同规模多完整副本的显存 | 运行开销抵消价值，或显存减少仅来自未驻留全部专家 |
| H4 | 可部署的请求级路由能获得部分 oracle 收益 | 路由误差或路由开销使实际收益消失 |

这两个预实验用于发现问题和确定技术路线，不直接构成两个创新点。TaskSpec 已使用任务特定 LoRA 草稿；MetaSD 已研究多草稿选择。因此不得声称“首次多 LoRA 草稿”或把已有工作的功能简单组合称为创新。

## 1.2 硬件、边界与本轮执行范围

硬件固定为 **单张 RTX 6000D / 84 GB**，Linux 租赁实例。租赁费用、运行时间和实际 GPU 可用显存由启动时记录，本文不假定本地 Windows 工作区已经连接该实例。

- 所有被当前配置使用的专家权重常驻同一张 GPU；无 CPU offload，无在线专家权重淘汰。
- 每个请求选择一次专家，整个回答保持该专家；请求之间清空 KV，保留权重。没有请求内 LoRA 切换，也不把旧专家 KV 传给新专家。
- 目标与草稿在同卡顺序执行。此阶段只测本地参考投机解码；网络延迟设为零，异步分支关闭。
- 本轮不训练 RL、不搜索网络策略、不声称真实云边部署加速、不把 6000D 耗时当成 3090 耗时。
- 只运行一个实验进程，不并发训练或后台占用 GPU。单配置一个新进程，避免其他模型残留改变显存结果。
- 所有配置必须报告真实显存，不强迫不同规模模型显存相等；另按最终边缘 24 GB 预算检查可行性。边缘预算只包括草稿侧，不包括真实部署中位于云端的目标模型；物理 84 GB 不等于允许最终边缘方案使用 84 GB。
- 冷加载只记启动日志，不作为主比较指标；全部 GPU 驻留后，不能人为给完整模型基线插入换入延迟。

先读 `nvidia-smi`，保存 GPU 名称/UUID、总/空闲显存、驱动、功耗上限、温度、时钟、CPU/RAM、容器镜像与是否共享实例。不同 GPU/功耗限制的结果不能混为一组。

## 1.3 软件与模型配置

| 项目 | 固定配置 |
|---|---|
| 目标模型 | `Qwen/Qwen3-8B`，冻结全部参数，禁止训练或更换目标 |
| 必做草稿 | `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-4B` |
| 可选中间规模 | `Qwen/Qwen3-1.7B`，仅在两个端点的结果显示需要定位折中点时补做 |
| 权重/计算精度 | BF16；不使用 4/8 bit 量化、QLoRA、CPU offload |
| 解码 | `do_sample=False`；非 thinking；不启用 top-k/top-p 筛选、重复惩罚或其他 logits processor |
| 聊天模板 | 统一 system 文本 `You are a helpful assistant.`；统一目标 tokenizer 的 chat template，`enable_thinking=False` |
| 推理 batch size | 1 |
| 注意力 | Transformers `sdpa`；固定实际可用后端并记录；所有组保持一致 |
| 编译 | 初版关闭 `torch.compile` 和 CUDA Graph，避免不同编译投入造成偏差 |
| 运行库 | PyTorch 支持 Blackwell 的 CUDA 构建；建议从 cu128 或更高的兼容平台镜像起步 |
| Transformers | 至少支持 Qwen3（官方说明为 4.51.0 起）；实际使用版本在冒烟测试后精确锁定 |
| 训练工具 | Transformers + PEFT + Accelerate；AdamW；不依赖第三方训练服务 |

安装时选择相互兼容的稳定版本，不把“最低版本”误认为已验证的完整依赖组合。环境冒烟通过后保存 `pip freeze`、容器标识和所有模型/数据 revision SHA；正式批次中禁止升级包或跟随可变的 `main` 下载。

必须检查各模型 tokenizer 的词表 ID、特殊 token、模板及 token 编码是否兼容。统一输入 token IDs 之前进行断言；“同属 Qwen 家族”不替代检查。首版使用目标 tokenizer，草稿也必须通过该兼容性检查。

## 1.4 数据集、固定划分与去重

以英文三个领域起步，减少语言与领域共同变化。数据筛选、划分种子统一为 **20261015**，与训练随机种子分离。训练和调参只使用训练池，最终测试锁定后不得反复查看用于选方案。

| 领域 | 数据源与字段 | expert_train | adapter_val | router_train | router_val | 最终 ID 测试 |
|---|---|---:|---:|---:|---:|---:|
| math | `openai/gsm8k`，`main/train`，只取 `question` 作输入 | 2000 | 200 | 400 | 100 | 同一 train 池另留 300 |
| code | `ise-uiuc/Magicoder-OSS-Instruct-75K`，`train`，`lang=python`，只取 `problem` 作输入 | 2000 | 200 | 400 | 100 | 同一池另留 300 |
| general | `HuggingFaceH4/ultrachat_200k`，`train_sft`，只取首次 user 消息 | 2000 | 200 | 400 | 100 | 从 `test_sft` 独立抽 300 |

math/code 各需去重过滤后 3000 条，general 的 train_sft 需 2700 条。各列互不重叠；通用 LoRA 训练集为三个 expert_train 的并集，共 6000 条。

外部泛化测试仅在方案冻结后运行：GSM8K 官方 `main/test` 固定抽 300 条；MBPP `sanitized/test` 固定抽至多 200 条；HumanEval 全部 164 条作为可选补充。它们不参与训练、路由标签生成、checkpoint 或草稿长度选择。MBPP/HumanEval 只提供题面/函数签名，不把答案或测试用例放入提示。

处理规则：

1. 使用 source ID 保留原始溯源；UltraChat 按 `prompt_id` 分组，不拆开同一对话产生训练/测试泄漏。
2. 用归一化题面 SHA256 去精确重复，做近重复/代码相似检查；外部测试重叠只据输入移除训练样本，不能据测试模型表现筛选。
3. 使用固定目标 tokenizer 过滤 prompt 长度，主实验保留 **不超过 1024 token** 的输入；不为某组单独截断。general 保留原始领域混杂，记录类型，不根据某个专家效果人工挑样本。
4. 过滤后再按固定种子划分；记录每个阶段被排除的数量/原因。若不足预定数量则记录实际数量，并在协议更新后统一调整全部组，禁止重复抽样伪造数量。
5. data manifest 保存 dataset ID/config/revision/split/source ID、domain、partition、prompt hash、token 数及去重策略版本。
6. 从 expert_train 各取前 64 条做 GPU 冒烟，不使用最终测试集；这不是独立的测试证据。

上述计数是实验设计目标，不是声称已经下载或筛选完成的数据数量。

## 1.5 目标模型生成训练轨迹

为草稿与目标对齐，所有组训练于同一批目标模型回答，而非各自生成不同数据。

1. 加载冻结的 Qwen3-8B BF16，使用 1.3 的统一模板、非 thinking、greedy；训练轨迹 `max_new_tokens=512`，遇 EOS 正常结束。
2. 对 expert_train 与 adapter_val 生成回答并保存；router_train/router_val 的目标轨迹可在路由阶段再生成，绝不并入适配器训练。
3. 保存完整 prompt/response token IDs、EOS 状态、是否达到长度上限、模型 revision、generation config hash。达到上限的有效回答可以保留，但不得伪造 EOS。
4. 不按数学是否答对或代码是否通过来筛选，只排除空输出、执行失败和数据格式错误；这里学习的是固定目标行为，不是另外筛选高质量教师。
5. 输入最多 1024，回答最多 512，总长应小于 2048；保存实际长度，不在不同模型训练时重新生成。
6. 卸载目标模型，分别训练草稿适配器。教师生成、数据准备和训练时间单列，不计入在线推理速度。

首版是 **sequence-level distillation / assistant-only CE**，不是完整概率分布 KL 蒸馏。不能把二者混称。若后续引入 KL，需独立实验、相同基线预算及明确教师概率存储成本。

## 1.6 LoRA 训练参数与 checkpoint 选择

以下参数是预实验起点，不是预先证明最优的参数。

| 参数 | 固定值 |
|---|---|
| 方法 | 标准 LoRA，冻结主干；每个领域从相同原始主干独立初始化 |
| target_modules | `q_proj`、`v_proj`，所有 Transformer 层 |
| rank / alpha | r=16，alpha=32 |
| dropout / bias | 0.05 / `none` |
| task_type | `CAUSAL_LM` |
| loss | 仅目标回答 token 的 next-token CE；system/user/padding labels=-100；正确处理 shift |
| optimizer | AdamW，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01 |
| scheduler | cosine，warmup_ratio=0.05，max_grad_norm=1.0 |
| epoch | 2；不提前对某一组增加 epoch |
| micro batch / accumulation | 1 / 16；有效 batch=16 个样本 |
| max_seq_length | 2048 |
| packing | 关闭，不拼接不同样本，不允许跨样本 attention |
| memory | gradient checkpointing 开启，训练 `use_cache=False` |
| precision | BF16；记录实际可训练参数 dtype、优化器状态 dtype |
| seeds | 初筛 42；关键比较复验 42、43、44 |
| 保存 | epoch 1、epoch 2 各保存一次；完整保留训练日志和实际更新步数 |

每个规模分别训练 math/code/general 三个领域 LoRA 和一个混合领域 LoRA。混合数据按领域均衡打乱；三专家每个看 2000 条两遍，混合专家看 6000 条两遍，**总体源样本及训练 token 暴露量相同**。若均完整处理，三专家合计约 750 次 optimizer update，混合专家约 750 次；记录实际步数及非 padding token，不能把“每专家相同 epoch”误写为每组相同 FLOPs。

参数预算对照：在核心比较中补做单个混合领域 **r=48、alpha=96** LoRA，其总低秩参数约等于三个 r=16 适配器之和。其他参数相同，记录真实参数数；r=48 的活跃计算量不同，不能声称完全计算匹配。

checkpoint 选择只用 adapter_val：固定 L=8，在目标 greedy 参考轨迹的公共前缀上计算平均连续接受长度。领域专家按对应领域选择，混合专家按三领域宏平均选择；相同时选较早 epoch。CE 作为诊断记录，禁止依据最终测试集选 epoch。选完后冻结权重。

默认不做学习率大搜索；若所有组发生不收敛/NaN，先排查实现与精度。必要时统一改 lr=5e-5 并提升协议版本，重跑受影响比较，不能只给候选额外调参。

## 1.7 对比组、最小运行矩阵与公平性

对两个必做规模 s∈{0.6B,4B} 使用以下命名：

| ID | 配置 | 回答的问题 |
|---|---|---|
| AR8 | Qwen3-8B 单独 greedy 自回归 | 目标输出与速度基线 |
| BASE-s | 原始草稿，无 LoRA | 微调是否有效 |
| MIX16-s | 混合领域单 LoRA，r=16 | 同活跃 LoRA 规模的通用基线 |
| MIX48-s | 混合领域单 LoRA，r=48 | 排除仅保存更多适配参数的解释 |
| E-MATH/CODE/GENERAL-s | 三个领域 LoRA 分别固定使用，全部驻留 | 专家×任务互补矩阵 |
| DOMAIN-s | 已知数据领域选对应专家，请求级固定 | 领域先验参考；不是实际路由准确率 |
| FIXEDBEST-s | 用 adapter_val 选出一个固定专家，跨任务使用 | 排除某一个专家全面领先 |
| ROUTE-TASK-s | TF-IDF 领域分类器选择专家 | 可部署的任务检测路由基线 |
| ROUTE-ACC-s | 以接受长度最佳专家为标签的同结构分类器 | 可部署的接受反馈路由候选 |
| ORACLE-s | 离线枚举三专家得到的接受质量上限 | 互补的潜力；禁止当在线算法/实际吞吐 |
| FULLCOPY-0.6 | 同一组三个 0.6B 领域 LoRA 分别合并成三个独立完整模型，全部驻留 | 与4B共享专家比较规模/部署权衡 |
| FULLCOPY-4B | 同一组三个4B领域LoRA分别合并，全部驻留；仅用短子集测部署成本 | 隔离“共享主干”对显存和执行的影响 |

“完整模型”表示独立完整权重副本，**不表示做了全参数微调**。合并副本与共享版本来自同一适配器，避免训练差异干扰部署比较；合并的数值差异要检查。真正全参微调0.6B属于可选额外基线，不放进首轮必做矩阵。

三完整0.6B与4B共享主干不是同显存配置，要画实际显存—质量—速度曲线，不宣称后者一定省显存。对不同配置，各自只驻留其规定的模型，不把所有规模混放在一个进程。

训练量：两个规模 ×（3领域＋MIX16＋MIX48）=10次适配器训练/seed，合并副本不额外训练。先做 seed42；关键结论需把候选及其比较基线一起补 seed43/44，不只重复赢家。1.7B暂不进入必做矩阵。

## 1.8 预实验一：主干规模与专门化的收益—成本曲线

**MOTIVATION：**更大的共享主干可能提供更长接受前缀，但更慢的草稿生成会抵消收益；同时要区分主干变大与领域专门化两种来源。

流程：

1. 在三领域 adapter_val 上跑 BASE、MIX16、MIX48、DOMAIN，L∈{4,8,12}。
2. 固定 L=8 给出主图，避免不同组使用不同长度掩盖差异。
3. 每个方法另外按验证集选择自己的 L*，评价表增加“各自调优后”列，禁止只给候选调优。
4. 冻结 checkpoint/L* 后，在最终 ID 测试三领域各300条上评估；正式每条生成至EOS或256个新token。
5. 做两个分解：同规模 MIX→DOMAIN 的收益；同结构0.6B→4B的收益。报告 FULLCOPY-0.6 与共享4B的实际显存，不以专家数乘主干计算活跃 FLOPs。
6. 若4B接受更长但本地投机更慢，保留这个结果；不得偷偷改变目标模型来制造优势。它可能是第二点隐藏草稿成本的动机，但尚不能证明异步一定能弥补。

本轮主结论来自本地实测。若后续做网络轨迹重放，另标“模拟网络结果”，不可混入本轮实测主表。

## 1.9 预实验二：专家互补、路由空间与同驻留部署成本

**MOTIVATION：**领域标签不一定对应最易被目标接受的专家；即使存在互补，也要证明轻量选择能利用它，且不是由一个万能专家或额外参数独占收益。

公共前缀诊断：

- 对同一提示预生成冻结目标的 greedy 参考回答，最多256 token。
- 从回答位置 j∈{0,32,64,128,192} 中选仍有至少 L+1 个参考token的位置。每个专家从**同一完整前缀**生成 L 个候选，统计首个不匹配前的连续接受长度 a∈[0,L]。
- 草稿候选必须自回归生成，不能用逐token teacher forcing 的命中率冒充整段接受长度。EOS边界不足的窗口排除并单列数量；短回答没有合格窗口时报告覆盖率，仍保留该请求在线时延结果。
- 每请求先对其合格窗口平均，再跨请求宏平均；bootstrap以请求为单位，不能把同一请求多个窗口当独立样本。
- 保存每专家 a 的分布，以及 S_e(k)=Pr(a≥k)，k=1…L。S_e(k)在此是当前greedy协议和数据上的经验生存曲线，不是sampling的理论接受概率。
- ORACLE按每请求三专家公共窗口平均 a 取最大；这是使用未来参考答案的离线上限，不给其报告可部署的在线 tokens/s。

实际路由：

1. 输入只用生成前的 user prompt，不读取测试答案、目标 logits 或未来接受反馈。
2. TF-IDF：word ngram=(1,2)，max_features=20000，min_df=2，sublinear_tf=True；只在router_train拟合。
3. 分类器：L2 LogisticRegression，C=1.0，max_iter=1000，class_weight=balanced，seed=42，CPU执行；固定 sklearn 版本。
4. ROUTE-TASK 标签是数据源领域；ROUTE-ACC 标签是router_train公共前缀测出的最佳专家，L=8；并列优先领域对应专家，再按math/code/general固定次序。
5. 两类路由器分别用router_val报告领域准确率、最佳专家命中率和选择后的接受损失，固定参数后再开测试集。不同草稿规模的接受标签分别生成。
6. 报告三专家各自选择比例。路由真实延迟包含文本特征提取、分类和专家激活，而非只计分类器 `.predict()`。
7. 在同驻留条件下测共享与合并副本的请求级选择＋prefill＋首草稿token耗时；初次模型加载单列，不作为在线主要优势。

先用validation估计oracle空间。若收益极小，可停止复杂路由投入；固定配置的测试仍可完成以报告否定结果。此阶段不加请求内切换/新KV共享设计。

## 1.10 投机解码正确性与实现约束

采用统一参考引擎实现 greedy draft→verify：目标按顺序接受与自身argmax相同的草稿前缀；首个不匹配处使用目标token；全部接受时按标准规则补充目标token；统一处理EOS与256-token输出上限。

- 目标KV及当前草稿KV跨轮保留，只回滚拒绝后缀；禁止每轮重做整个prompt prefill。
- 所有方法共用同一验证、缓存管理、停止规则和目标精度，不能用慢的自写AR对比特别优化的候选。
- 每个请求目标输出与同一精度/后端下 AR8 的完整 token IDs 比较；正确性 mismatch_rate 必须为0，才允许进入速度主表。
- 冒烟至少覆盖首token拒绝、部分接受、全部接受、EOS落在草稿中、输出上限、不同prompt长度。实现这些必要的解码正确性检查；不为文档编辑编造GPU测试通过结果。
- BF16批量验证与逐步AR可能因数值细节产生argmax差异；一旦发生须记录最早分歧与logits差值并排查，不能删掉失败样本、放宽正确性要求或把不一致称为无损。
- 合并模型与未合并模型也检查logits/greedy输出差异；若无法保持同一草稿行为，部署对照注明并报告，不能声称严格等价。
- sampling、概率接受校正及Top-K通信压缩不在此阶段；未来加入必须另建协议与分布正确性检查。

## 1.11 必须记录的指标与精确定义

记第r轮草稿实际长度 L_r，接受草稿数 a_r，最终新确认数 n_r；n_r包括目标纠正/bonus，截断处按实际交付计数。

| 指标 | 定义/单位 | 用途 |
|---|---|---|
| draft_acceptance_rate | Σa_r / ΣL_r | 接受质量，排除未实际生成候选 |
| mean_accepted_draft_length | Σa_r / round_count | 只含草稿接受，不含目标补充 |
| mean_committed_per_round | Σn_r / round_count | 对应实际每轮进展 |
| accepted_length_hist / S_e(k) | 分布及Pr(a≥k) | 专家互补、后续长度控制的输入 |
| local_request_latency_ms | 已tokenize输入就绪后，路由开始至最终确认输出完成；包含H2D、prefill、生成、验证、缓存处理 | 本地完整推理耗时；不含下载、启动、tokenization、字符串后处理 |
| local_confirmed_tokens_per_s | Σ请求最终确认token / Σ请求本地完整耗时 | 单请求顺序处理下的汇总有效速度，不称多用户容量 |
| local_decode_tokens_per_s | Σ(N_i−1)/Σ(t_last−t_first)，只对N_i>1请求 | 首token之后速度，和完整请求速度分开 |
| local_speedup_vs_AR | ΣT_AR8 / ΣT_method，相同请求及相同输出 | 实测加速比；EOS使长度不同的错误输出不允许参与 |
| TTFT_ms / TPOT_ms | TTFT=t_first−t_start；TPOT=(t_last−t_first)/(N−1) | 首token与平均间隔，给出请求中位数/P95 |
| routing_ms / prefill_ms | 完整路由/两模型prefill分别计时 | 确认开销是否吞噬收益 |
| draft_ms / verify_ms / cache_management_ms | 每轮/每请求分别记录 | 时间分解；必须说明未覆盖时间及同步成本 |
| target_forward_calls / rounds | 实际计数；prefill和verify分开 | 判断是否减少目标调用 |
| weights_bytes / adapter_bytes | tensor numel×element_size，去除真正共享的存储重复计数 | 真实权重与增量大小；并给GiB |
| peak_allocated / peak_reserved / NVML_used | 峰值GPU allocated/reserved、NVML进程占用，分别报告 | 完整峰值；同时分离目标和草稿侧估计 |
| trainable_params / training_tokens / steps / seconds | 每adapter及整组总计 | 对比训练投入，不混称计算公平 |
| output_mismatch_rate / failures / OOM | mismatch率、失败原因和数量 | 正确性与可行性门槛 |

跨领域比较：每个领域分别报告 R_d、speedup；主选择分数为三领域加速比几何平均：

`macro_speedup_geo = exp(mean_d(log(sum_T_AR_d / sum_T_candidate_d)))`，higher is better。

另报告各领域平均接受长度及差值。不能把逐请求速度平均、总token/总时间、平均TPOT的倒数混为一个指标。

主分数是本地速度的筛选信号，不取代接受质量/显存诊断；完整方法是否改善真实云边SLO要在第二点验证。第一点不捏造SLO阈值、不宣称SLO已满足。

## 1.12 计时、显存与重复策略

1. 训练后执行 `eval()`，关闭dropout、梯度；生成在 `inference_mode`。每配置先10个非测试热身请求，不计入结果。
2. 完整请求使用 `perf_counter` 配合计时边界 `torch.cuda.synchronize()`；CUDA Event测GPU组件，CPU路由用墙钟。详细profiling与正常吞吐分开运行，避免每轮强制同步人为降低速度。
3. 请求之间清空KV和会话状态，不复用跨请求prefix cache。不要每轮 `empty_cache()`；显存统计按新进程和明确reset点执行。边缘显存可行性另开仅含草稿主干、该配置全部专家和草稿KV的进程测量，不能从两个不同运行的总峰值简单相减来估算。
4. 同seed权重的完整评测重复3次；随机化配置运行顺序、固定样本顺序，记录温度/功耗/其他进程。评测重复不是独立训练seed。
5. 报告各次运行和中位结果；质量差值与时延比使用按请求配对bootstrap（2000次，95%CI）。多个训练seed另外报告均值/标准差，不混淆两种不确定性。
6. 组件微基准上下文256/1024/2048；4096为可选扩展。从固定真实token前缀库取样，不用相同token无限重复；每点预热10次、测量100次，报告中位数/P95。
7. 记录prefill和稳态decode分开。模型首次加载/编译不计在线速度，但启动时间单独保存。
8. 若GPU OOM，先确认无额外模型残留；OOM本身保存为结果。不能只为候选缩短上下文或偷偷量化。需变更协议时所有受影响比较统一重跑。

## 1.13 从环境检查到正式结果的执行顺序

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| P0 | GPU/PyTorch/BF16/SDPA、模型前向、LoRA梯度与tokenizer检查 | 64条冒烟跑通，实际显存、依赖锁和模型revision已保存 |
| P1 | 去重划分与目标轨迹生成 | 数据manifest冻结，split无交叉，教师输出可复现 |
| P2 | seed42训练两个规模各5个适配器，保存合并副本 | 每组训练token/步数可核对，val checkpoint选定 |
| P3 | validation接受质量矩阵、L扫描、部署微基准 | 明确主干与多专家是否有空间，冻结L*与候选 |
| P4 | 存在路由空间时训练两个轻量router | 无测试输入/答案泄漏，router_val完成，CPU耗时记录 |
| P5 | 固定候选与完整比较基线补seed43/44 | 多seed结果与初筛结论一致或明确报告不一致 |
| P6 | 解锁最终ID测试、外部泛化测试 | 正确性通过后生成正式主表；测试结果不再反向调参 |
| P7 | 整理图表与结论、导出第二点可用的profile | 原始日志、配置和图表可由相同命令再生成 |

成本控制：先用P0的64条发现执行错误；所有配置的正式数量保持同一协议。先端点再中间规模；不承诺固定小时内完成，因为尚无此卡实际训练/生成速度测量。

## 1.14 autoresearch 执行契约与停止规则

已安装第三方Codex skill：`uditgoenka/autoresearch`，版本以安装清单和本地 `SKILL.md` 为准。下一轮可通过 `$autoresearch` 选择；它不等于已经配置好远程GPU或已经运行训练。

原版Karpathy的 `train.py` / `prepare.py` 不能直接完成本实验。当前目录也不是Git仓库；未来执行循环时，在GPU实例建立专用实验仓库。以下为未来实现接口约定，**当前工作区尚未提供这些训练/评测脚本，不得把示例命令当作已验证可运行**：

```text
prepare_data.py       固定数据/划分/教师轨迹与manifest
train_lora.py         允许修改的训练实现，配置必须另存
eval_sd.py            固定投机评测与正确性检查
profile_models.py     固定计时/显存口径
train_router.py       固定两种路由基线
configs/              可变候选训练参数，evaluation配置冻结
reports/              从原始结果生成图表
```

未来GPU阶段的classic配置（前置脚本实现并实际dry-run后才能启用）：

```text
$autoresearch --classic
Goal: 在固定数据、Qwen3-8B目标与驻留假设下，提高验证集macro_speedup_geo并解释质量/计算/显存变化
Scope: train_lora.py, configs/candidates/*.yaml
Metric: macro_speedup_geo
Direction: higher_is_better
Verify: python eval_sd.py --config configs/eval_val_frozen.yaml --metric-only macro_speedup_geo
Guard: python eval_sd.py --config configs/correctness_frozen.yaml --check-correctness
Iterations: 6
```

实际执行前，必须在GPU实例完成Verify/Guard的dry-run并得到有限数值与正确性结果；缺少脚本/数据/GPU时记录 `NOT_RUN` 或 `blocked_prerequisite`，不得填模拟数字。初版协议冻结前的开发修复不是训练搜索。

每次只改一个逻辑因素，例如rank或lr。禁止更改冻结目标、tokenizer、测试集、接受规则、计时区间、L选择程序或删掉失败请求来提高分数。原始日志和失败checkpoint索引保留；不覆盖最好结果，不自动发布、推送或创建定时循环。

每次训练run独立目录，不用破坏性Git reset清理实验。在干净的专用实验仓库使用可回退提交；共享工作区不执行清空工作树命令。当前用户只要求制定方案，不触发上述6轮训练。

记录到 `autoresearch/loop-YYMMDD-HHMM/`：

```text
results.tsv: iteration,timestamp,run_id,config_sha,commit,seed,scale,method,rank,L,metric,delta,guard,status,description
requests.jsonl: run_id,prompt_id,domain,N,total_ms,ttft_ms,rounds,accepted,drafted,peak_allocated,peak_reserved,token_match
rounds.jsonl: run_id,prompt_id,round,expert,L,a,n,prefix_len,draft_ms,verify_ms,cache_ms
train.jsonl: run_id,adapter,step,epoch,loss,lr,trained_tokens,peak_memory,wall_seconds
profiles.jsonl: model,revision,expert,context_len,L,operation,time_ms,repeat,backend,precision
handoff.json: protocol_version,status,config_hash,results_paths,findings,next_steps
```

上面的逗号分隔只是字段列表；`results.tsv`实际使用tab，JSONL一行一个对象。尚无结果时metric留空/null，不能写0表示未测。

保留/淘汰：正确性失败直接crash；验证指标变差discard；小于计时不确定性的变化记inconclusive并复测。6轮上限到达即停。耗时超预算或连续2次同类OOM/NaN时保存状态、停止该批次并诊断，不无限重试。

## 1.15 预先登记的判断标准与衔接第二点

以下5%是管理预实验投入的启发式门槛，不是文献公认阈值；最终结论同时报告效应大小和置信区间。

- **多专家继续门槛：**validation上ORACLE相对MIX16平均连续接受长度提升至少5%，并对MIX48给出差值。达不到时先考虑改善训练或减少专家，不直接开发更复杂调度。
- **实际价值：**可部署路由相对同规模MIX16/MIX48的接受收益和macro_speedup_geo一起报告；接受率改善而本地速度下降只称质量—计算权衡，不称整体加速。
- **较大主干：**4B若比0.6B质量更好、计算更慢，画Pareto曲线并保留候选；是否适合异步隐藏成本交给第二点，不提前判定4B必胜。
- **共享部署：**与同规模FULLCOPY比较实际权重/峰值显存、prefill及稳态耗时；与不同规模FULLCOPY-0.6比较时明确资源不同。
- **否定结果：**oracle弱、router未超过通用LoRA、4B不划算、某个单专家占优，都是可报告结果。不得用合成“跨领域突变”样本强行证明请求内切换必要。

本轮输出第二点需要的真实profile：`S_e(k)`、接受长度直方图、不同上下文/L下草稿/验证时间、路由与prefill时间、显存。第二点再做RTT/带宽/异步分支扫描；只有出现专家接受曲线交叉或异步价值排序差异，才研究网络约束下联合选专家。所有专家成本和载荷相同时，固定L下增加共同通信延迟并不会自动改变最佳专家。

## 1.16 交付图表与实验完成清单

必需图表：

1. 专家×任务平均接受长度热力图，含混合领域单LoRA。
2. 0.6B与4B：接受长度—草稿耗时—本地速度对比，L=8与各自L*分开。
3. BASE/MIX16/MIX48/FIXEDBEST/DOMAIN/ROUTE/ORACLE质量对照；oracle不画在线吞吐。
4. 共享LoRA与同规模合并完整副本：权重、峰值显存、完整请求耗时对照。
5. 三专家接受长度生存曲线，带请求级不确定性。

完成清单：

- [ ] 环境与revision锁定，manifest和split检查通过。
- [ ] seed42必做训练与正确性检查完成。
- [ ] validation完成，checkpoint/L*/候选固定。
- [ ] 关键候选及基线完成三个训练seed。
- [ ] 最终测试、外部泛化和失败记录齐全。
- [ ] 图表可从原始日志再生成，实测/估算/未来实验分开标注。
- [ ] 交付一页结论：支持/否证哪些假设，第一点仍缺什么方法创新，第二点应检验什么。

上述复选框当前均未完成；文档存在不等于实验已完成。

## 1.17 来源、版本依据与适用范围

以下来源用于核查现有方法、模型接口和数据结构；本文的样本数、超参数、实验门槛是针对本项目制定的方案，并非声称来自这些论文的最优配置。

- autoresearch skill：[uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch)，Codex源目录 `.agents/skills/autoresearch`；原始范式：[karpathy/autoresearch](https://github.com/karpathy/autoresearch)。
- 模型：[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)、[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)。
- 数据：[GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[Magicoder OSS Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)、[UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)、[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)、[HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)。
- 软件：[PyTorch 2.7 Blackwell/cu128支持说明](https://pytorch.org/blog/pytorch-2-7/)、[PEFT LoRA官方文档](https://huggingface.co/docs/peft/en/developer_guides/lora)。
- 相关工作：[TaskSpec](https://arxiv.org/html/2505.08600v1)、[MetaSD](https://arxiv.org/html/2604.05417v1)、[BanditSpec](https://proceedings.mlr.press/v267/hou25h.html)、[HedgeSpec](https://arxiv.org/html/2510.20064)、[S-LoRA](https://arxiv.org/abs/2311.03285)。

文档的本地耗时不能直接推广为3090边缘或真实WAN性能；可部署性结论需后续在3090补profile，云边加速与SLO结论需第二点的独立实验。

---

## 2. 第二版：五任务共享 LoRA 的反馈与成本约束路由（2026-10-17）

本节是**新实验协议 v2.0**，不是对第一版结果的改写。用户授权在独立 `v2` 目录运行 autoresearch、自主修复报错并每 15 分钟检查，直至本节有限清单完成。第一版原始程序/数据/结果不可覆盖；没有实测的格子保持“未测”。目标是检验可以部署的**净收益**，无正收益也算有效的否定结果，不准借调整测试集、正确性或计时口径把它改写为成功。

### 2.1 第一版实测结论与第二版的起点

证据：`reports/continuation-20261017-final_report.json`，远端 `results/continuation-20261017/final_report.json`。第一版完成 8100 条共享教师轨迹、两个草稿规模各五个 seed42 LoRA 的两轮训练/选模、L=8 两规模各 600 请求 × 六草稿质量矩阵、两规模各 1500 条接受路由标注、每规模 24 条 × 七方法的一次在线诊断。在线只有 48 条独立 AR 计时、336 次 SD 比较，不是原计划的完整在线测评。最终 ID/外部测试、seed43/44、L4/L12 扫描、真实 WAN 与长时稳态测量尚未做。

| L=8，600 条验证矩阵平均连续接受草稿长度 | 0.6B | 4B |
|---|---:|---:|
| 无 LoRA BASE | 3.746 | 5.192 |
| 混合 MIX16 | 4.251 | 5.472 |
| 参数预算混合 MIX48 | 4.290 | 5.471 |
| 按已知领域 DOMAIN | 4.228 | 5.451 |
| prompt-only ROUTE-ACC | 4.227 | 5.434 |
| 逐请求事后 ORACLE | 4.508 | 5.736 |

接受质量矩阵显示两规模存在专家互补，但 ROUTE-ACC 对 MIX48 分别低 0.064、0.037 token；领域识别约 99%，接受最优专家命中约 70–72%。ORACLE 对 MIX48 的 0.218/0.265 token 差值是固定专家池的**事后质量空间**，不是可实现速度、也未扣除选择开销。原同卡参考实现各配置在 24 条单次计时上仍慢于 8B 直接 AR，因此仅凭更长接受前缀不能声称提速。4B 共享主干和三个 0.6B 完整副本不等显存；合并 BF16 副本曾出现数值/输出差异，不报严格等价。以上只支持 seed42 验证阶段结论。

第一版的 `general` 取自通用指令数据，不再作为第二版的“第五个专家”。它仍可作固定通用草稿诊断，但第二版主比较用同五任务数据训练的混合 LoRA。

### 2.2 前人工作、问题与预注册假设

- [TaskSpec](https://arxiv.org/html/2505.08600) 已用任务专门草稿和 prompt 分类路由，并比较混合单草稿；“有领域专家”“识别语义领域”不是创新。
- [MetaSD](https://arxiv.org/html/2604.05417) 已用目标验证反馈逐轮选择多个草稿，并在附录讨论切换成本；不能将“反馈路由/代价”笼统写成首创。
- [SpecRouter](https://arxiv.org/html/2505.07680) 已将运行时间、对齐和动态推测配置纳入路由。只换一个目标函数不足以宣称新方法。
- [E²-SCI](https://openaccess.thecvf.com/content/CVPR2026/html/Li_E2-SCI_Elastic_Edge-Cloud_Speculative_Decoding_via_Credit_Inertia_CVPR_2026_paper.html) 处理边云验证阈值与流水线，不是多专家选择；它报告保持任务准确率，不等于本实验要求的逐 token greedy 完全一致。

**主假设 V2-H1：**在同等数据/参数与固定目标下，五个专家存在稳定的逐请求或请求内互补，而不仅仅是事后取最大值造成的表面增益。**V2-H2：**结合提示词先验与已观察的验证反馈，比 TaskSpec 风格静态选择和 MetaSD 风格反馈选择更准确地预测当前草稿的*净已确认 token/时间*；实际切换、KV、路由与通信成本扣除后仍有收益。**V2-H3：**共享 LoRA 的权重驻留/切换特性在受限显存和边云时延下产生与独立完整草稿不同的可测权衡。三条均可被否证；目前没有成立的“独特路由优势”。

第二版候选策略只在有条件时探索：提示词给初始专家；每轮利用已发生的验证反馈更新收益估计；只有预计额外收益超过实测 LoRA 切换、KV 补建和额外验证成本才切换或改变 L；预测不确定时保留当前专家/退回 AR。每项组件与强基线做消融，不能仅将既有方法拼接就称创新。在线真值不可使用未来目标 token；Oracle 独立列为事后诊断。若研究请求内切换，第一版“逐请求 Oracle”不再是该策略的数学上界，需要另建相同状态空间的离线参考并清楚标注其信息特权。

### 2.3 五任务与固定数据边界

主任务与 MetaSD 的任务类型和公开评测对齐：代码、德英翻译、摘要、自然问答、数学。计划对应 MT-Bench code、WMT16 de→en、CNN/DailyMail、Natural Questions、GSM8K；当前可在磁盘预算内固定获取的问答来源是 `google-research-datasets/nq_open`，属于 NQ-Open 变体，**不等同于 MetaSD 所用的完整 Natural Questions 评测**。对它的结果必须标作近似任务对照；若后续能在存储预算内获取完全一致的 NQ，另列一组。MT-Bench code 请求少，只作外部小集，代码的统计主集另用 MBPP/HumanEval 严格锁定的题面。任务类型相同不等于跨论文速度可直接相除：模型、模板、输出长度、硬件和实现须在我们自己的基线间保持一致。

训练数据从公开许可且能固定 revision 的各任务来源选取；先记录原始来源、ID、许可、版本和模板，再冻结分区。不得将 CNN/DailyMail/WMT16/NQ/GSM8K 的评测项或相同题目改写版放入训练；做输入精确去重、归一化近重复检查及数据源分组切分。五任务各计划 `expert_train=2000, adapter_val=200, router_train=400, router_val=100, locked_ID=300` 条，训练与调参数据互斥；某官方集合不足 300 时使用实际唯一数量并预先记入 manifest，不复制样本凑数。数学、代码等原版数据只可在新分区无污染证据下重用。五专家合计训练曝光量与混合 LoRA 的数据总曝光量匹配；同时报告相同活跃 rank 与总参数匹配的混合 LoRA。每类输出长度及早停按同一目标轨迹记录，不能通过过滤难题提高指标。

正式下载和教师生成前先做**小规模可运行校验**：每任务固定 256 条 expert_train、48 条 adapter_val、96 条 router_train、24 条 router_val；这些是大集的预先固定子集，不从测试集按模型表现采样。小集仅用于正确性、吞吐和可分性预判，不作为论文主结果。给出实际采样数量、失败/排除原因与哈希后才能扩成完整协议。所有组用同一 Qwen3-8B BF16 非 thinking greedy 目标轨迹做 assistant-only 蒸馏；教师轨迹需要为**训练和静态质量比较**保存一次，在线 SD 的目标验证仍现场运行，不能拿离线教师结果冒充在线目标调用。可分片并行教师生产，按 ID 原子保存并全量核对去重与 EOS/截断。

### 2.4 模型、资源与机器分工

主研究先冻结 Qwen3-8B 目标；先测现有 0.6B 草稿与训练后的五 LoRA。4B 草稿只有在实测成本/质量预测显示可能超过 AR 或在独立边云流水线中隐藏草稿成本时才进入完整五专家训练；不能因期待更高接受率直接扩大。32B 目标另立扩展协议：模型/存储、显存、目标 AR、教师重生和比较基线全变，不能与 8B 实验混称复现 E²-SCI。

AutoDL RTX 6000D 负责冻结目标 8B 的严格正确性、质量矩阵和所有主速度计时。4090 主机已有独立任务/环境；可接收未分配 ID 的**教师生成或只需草稿的离线分片**，前提是该机空闲、有相同模型 revision/tokenizer/BF16 策略，写入隔离路径，导回时核对完整 token/EOS/截断和样本哈希，并在主设备抽样及边界重放。严格输出差异不能偷偷删样本；必要时该分片回 AutoDL 生成。跨机器的耗时不得合并成同一硬件加速比。分工前记录双方所有者和 ID 范围；第一版 4090 旧任务结果不得假装自动适用于第二版。

每一比较配置计时独占对应 GPU，记录设备型号/UUID/驱动/功耗、其他进程、峰值显存。AutoDL 只有约 10 GB 可用磁盘的当前事实必须进 preflight；满盘前先用新 v2 目录的容量估算和完整性策略解决，不删除第一版数据/模型或另一个聊天的环境归档。原 `spec_research`、tests、configs、第一版 `program.md` 哈希属于旧 P0 证据；v2 在独立工作树与 `spec` 环境创建新 gate，不改旧哈希伪装通过。

### 2.5 对照、指标和完成门槛

最少同条件比较：目标 AR、原始草稿、五任务混合 LoRA（活跃 rank 和总参数两个预算）、最佳固定专家、已知任务路由、TaskSpec 风格 prompt 分类、MetaSD 风格 UCB/验证反馈、候选成本约束路由、离线质量 Oracle。可忠实运行的 SpecRouter/E²-SCI 部分单列系统对照；无法同等复现的部分明示限制，不以不等价实现冒名。对每项保存确切训练数据量、训练 token、更新步数、权重显存和时延构成。

主要指标为同请求配对的已确认 token/s 或总时延比 AR，附 TTFT、p50/p95、每轮接受长度、路由/切换/KV/草稿/目标/通信时间、峰值显存与成本。质量矩阵可离线使用公共目标前缀，但主性能必须真实自回归草稿并现场批量目标验证。每次线上请求必须与同设备同精度目标 AR 的**完整 token ID、EOS 与截断**一致；不允许按失败结果删请求。Oracle 不报告在线吞吐。单卡和真实 WAN 分开；仿真 RTT 若做只能标为仿真。

检查点按顺序执行，后一项必须由前一项证据解锁：

1. **P0v2 环境/数据/解码 gate**：资产 revision、隔离目录、去重划分、正确性边界和实测成本可审计；记录干净 baseline。
2. **P1 五任务小集**：冻结小集 manifest、目标教师轨迹、0.6B 五 LoRA 与混合 LoRA；逐组检查实际曝光、NaN 和选模分区。质量矩阵与 Oracle 只作工作信号。
3. **P2 可分性 gate**：Oracle 对混合 LoRA 与最佳固定专家的配对差值及置信区间稳定为正，且预估收益超过至少一个可实现的路由/切换成本；若不成立，停止复杂路由，报告专家构造或数据不足，不偷看锁定测试来救假设。
4. **P3 路由/系统 gate**：先同设备比较静态 prompt、MetaSD 风格反馈和候选；检查请求内切换正确性以及分解后的净时延。若新方案不优于强基线或不能超过 AR，报告否定结果并分析是否为解码实现瓶颈；不把接受率当速度。
5. **P4 扩量与独立验证**：只有 P0–P3 通过才按已冻结数量扩展并补 seed43/44、至少三次顺序随机化计时与配对 bootstrap；一次性解锁测试、外部五任务和跨任务提示词，形成完整报告。若上游 gate 否证，生成有证据的停止报告，任务同样结束，不强制消耗完 GPU 预算。

成功谓词不是“研究想法必然成立”，而是 v2 报告完整写出每一 gate 的结果、未做项与理由、原始可重放记录、至少一个独立 holdout 正确性检查，并核实无自建残留进程。所有小样本结果只称诊断，不能写论文确定结论。

### 2.6 autoresearch 与 15 分钟值守

使用本机已安装的 `autoresearch` skill 的 **orchestrator** 模式执行上述有限目标；迭代记录放 `autoresearch/v2-routing-261017-*/`，包括 `orchestrator-state.json`、`results.tsv`、每轮 `handoff.json`、配置/源码哈希和保留/淘汰理由。默认最多 25 个有实测的变更循环；调度器可在跑长时训练/生成时等待，不把每次检查计作优化迭代。任何训练/路由超参变更只用工作集与 val，独立 holdout 复核后才接受；不通过的候选保留日志，回到先前最佳。不能为了分数更改目标、测试集、正确性、计时边界或删除失败。user 已授予实验执行与修复权限；skill 的发布/推送/部署禁令仍有效，本轮不自动上传 GitHub。

独立远端协调器维护原子状态、worker PID、分片 owner、心跳与恢复检查点；本聊天的 heartbeat 每 15 分钟读取进程/GPU/日志/数据计数/错误并自主修复可复现的编排问题。SSH 瞬断重试一次；重复 OOM 先降并发而不改模型精度/数据公平性，NaN 与输出分歧需实证定位，改受保护解码源码则重做 v2 P0。没有工作时可以报告空闲和原因，不为占用率空跑。正常检查静默，重大阶段完成、重大工期变化、无法修复或最终结束才通知。协调器退出时核对报告、进程、冻结文件与自动化状态，再停止 15 分钟值守。

## 3. 第三版：拒绝位置预测、预备分支与预算约束专家选择（2026-10-18）

本节是独立探索协议 v3.0，**不追认第二版 P3 已通过**。远端保留已受旧 gate 保护的根 `program.md` 原始 SHA256；本节的可执行副本、代码和结果写入 `experiments/v3-async-budget/`，与 v1/v2 分开。执行窗口以北京时间 2026-10-18 **17:20** 租卡释放为硬截止：17:00 前停止新 GPU 长任务，17:10 前终止本轮自建 GPU 进程并完成可复核的 GitHub 备份，17:20 前复查；用户若随后给出更精确的租卡信息，以实际较早时间为准。最多 8 次有实测的 autoresearch 变更循环，每 15 分钟检查；不为空跑而提高 GPU 利用率。

### 3.1 研究定位与可证伪假设

- [E²-SCI](https://openaccess.thecvf.com/content/CVPR2026/html/Li_E2-SCI_Elastic_Edge-Cloud_Speculative_Decoding_via_Credit_Inertia_CVPR_2026_paper.html) 发现相邻验证窗口的接受表现有惯性，并以 Progressive Lookahead Concurrency 异步隐藏边云验证等待。它不是我们“首次异步投机”的证据。
- [SpecBranch](https://arxiv.org/abs/2506.01979) 已在可能拒绝时提前展开分支；[SVIP](https://arxiv.org/abs/2411.18462) 已利用草稿不确定性自适应停止。单独预测拒绝位置、提前分支或调草稿长度均非首创。
- [MetaSD](https://arxiv.org/html/2604.05417) 已根据验证反馈选择草稿。我们需要检验的差别是：**在边侧显存、分支计算、上行字节和云端验证预算共同受限时，何时继续主草稿、在哪个可能拒绝位置准备几个纠错后缀、使用哪个常驻 LoRA 专家，是否带来严格正确的净时延收益。** 这只是待验证问题，不预设新颖性或正结果。

H3-1：只用当时可见的草稿置信度、此前验证结果、任务先验、上下文长度和成本状态，能够在按请求隔离的验证集上校准首拒绝位置的危险率，并优于固定长度、仅历史接受率及仅熵基线。H3-2：在相同算力/字节预算下，预测驱动的纠错分支相对不分支与均匀分支提高**目标实际确认 token/端到端时间**；若分支命中低或浪费高，保留否定结果。H3-3：预算约束的专家/长度/分支联合决策优于 MIX80、最佳固定专家、已知领域、MetaSD 风格反馈与 E²-SCI 风格惯性长度策略；严格输出一致率必须为 100%。

### 3.2 决策时序与正确性

每轮状态 s_t 只含请求开始前的 prompt 特征、已确认前缀、当前专家、前几轮接受数、草稿 token 自身 logits/entropy/margin、已耗边侧毫秒、驻留显存和剩余上行字节；**不得读取本轮目标 logits、目标纠错 token 或未来接受结果**。动作 a_t=(专家 e、草稿长度 L、候选拒绝位置 j、纠错 token 候选集合 B_j、各分支后续草稿长度)；动作也可为“不分支”或“直接目标 AR”。先由边侧异步起草/传输，云端只以冻结 Qwen3-8B BF16 greedy 验证；目标实际拒绝时，仅在其纠错 token 与某分支根一致且该分支上下文完全一致时复用该后缀，否则丢弃。所有输出仍按目标 argmax 提交，检查完整 token IDs、EOS 和截断，任何分歧立即停止候选计时，不删请求或放宽门槛。原版参考解码器不修改；新解码器与原版 AR 作独立逐请求比对。

收益为实际已确认 token 数/墙钟时间，次指标是 p50/p95/TTFT、接受长度、分支命中/废弃率、上行字节、草稿/验证/路由/传输分解时延及边侧峰值显存。Oracle 可事后选 j、纠错 token、专家，但只能作上限，不能报告可部署吞吐。若只有单卡串行运行，则只报“离线反事实”或“单卡参考实现”；网络 RTT/带宽重放必须标仿真，不能声称真实云边加速。

### 3.3 强基线、数据与预算

复用 v2 的固定目标/草稿资产和教师记录，只读；从 `router_train` 预先按请求 ID 哈希划分拟合 60%、调参 20%、锁定检验 20%，五任务分别分层。v2 已查看的 `adapter_val`/`router_val` 只用于探索性诊断，**不得成为 v3 独立证明**；如时间允许，另从未触碰的 locked ID 构造一次性最终集，必须先排除与所有旧 ID 的重复。训练标签可从相同目标参考前缀、全部专家的完整提案得到；任何反事实仅在该前缀有效，不能假装预知改写后路径。数据不足时只报告样本规模和不确定区间。

比较：目标 AR、MIX80 固定 L8、最佳固定专家、已知领域、仅历史接受惯性、仅草稿熵/长度、均匀分支、MetaSD 风格反馈，以及联合预算策略。每个比较在同一请求/目标/精度/输出上计时；网络方案分别扫 RTT {0,20,80} ms、上行 {5,50} MB/s 和边侧每轮草稿预算 {8,16} token-equivalent，分支数 {0,1,2}。这些值是**预注册仿真情景**，不是实测 WAN 条件；要采集真实网络须另有实际两端可用连接。分支 token、KV、副本、重发字节都计入预算，不能把多分支计算当免费。

决策可从代价约束上下文 bandit/短视距 MDP 入手：r_t=确认 token−λ_d·边侧计算毫秒−λ_u·上行字节−λ_v·云端验证毫秒−λ_s·切换/KV 重建毫秒，受显存、字节和延迟上限约束。先训练校准的拒绝危险率，再用经验时延表做约束搜索；仅在可重放环境通过独立 holdout 和真实性检查后试小规模在线强化学习。离线完整信息只用于候选排序，不能把同一轨迹上“模拟出来的策略收益”当真实在线速度。任何 RL 策略都必须同时报告简单贪心、UCB、固定策略以及成本消融；不以增加探索次数无限耗租卡。

### 3.4 有界预实验与停止 gate

P0：冻结 v3 manifest/代码哈希、GPU/驱动/依赖、旧 gate 状态和当前截止；核对样本无泄漏，建立直接 AR 与 v2 参考输出；所有实验 Python 用 `/root/miniconda3/envs/spec/bin/python`。P1：对上述分层请求收集逐轮首拒绝位置、草稿可见置信特征、实际成本；报告校准曲线/Brier、分位置 AUROC/PR、按请求 bootstrap CI 和不同任务表现。若优于“仅历史”或“仅熵”没有可重复信号，则停止昂贵分支，报告负结论。P2：只在 P1 有空间时，对同一前缀测试 0/1/2 分支的纠错根命中和后缀复用，统计计算/字节浪费；用**可部署特征**选分支，单独列事后 Oracle。若上界仍无法覆盖成本，则停止。P3：只有 P2 通过，才做逐请求严格正确的真实端到端解码和计时；先每任务 4 条冒烟，再每任务至少 8 条同请求配对，随机化方法次序；若任一 token/EOS/截断不一致则阻断速度结论。没有两端真实链路时，只能做标注清楚的 RTT/带宽反事实，不报告真实 WAN 提速。P4：只有 P3 净收益对强基线和 AR 均成立、置信区间与预算约束可审计，才计划额外种子或更大目标；本次租期**不自动扩量或改目标模型**。

每次迭代固定一个可测变化（危险率模型、分支根候选、阈值或专家动作），记录配置、源哈希、输入分区、GPU 用时、指标与 keep/discard；同一锁定集不反复调参。成功谓词是**出具 P0–P3 的通过/未通过/未运行清单和证据**，不是强迫获得加速。截止前如仍未完成，原子保存数据、checkpoint 和明确的未做项，停止本轮 GPU 进程，上传协议、有效代码、结果与失败诊断；不上传密码、密钥、基础模型权重、缓存或历史失败实现。GitHub 私有仓库 `ohhandme/sd-lora-preexperiments` 的新快照必须用提交/发布后的可读 URL 与哈希验证，不能把本地打包称作上传完成。

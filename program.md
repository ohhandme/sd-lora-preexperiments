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

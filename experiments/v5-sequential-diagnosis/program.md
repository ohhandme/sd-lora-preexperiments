# V5：依次诊断训练、互补、路由、结构

2026-10-18。用户明确授权按四步连续完成预实验，不在单项负结果处结束。四阶段均执行；负结果决定解释，不自动证明后一阶段有效。旧 v1–v4 代码、数据、权重、结果保持只读，不运行旧任务、不发布或推送。

## autoresearch

Mode classic，4个顺序阶段；Metric=选定失配池在本轮开发评估上的实际路由接受长度减旧MIX80（越大越好）；无指标时记NA而非0。Verify=`/root/miniconda3/envs/spec/bin/python experiments/v5-sequential-diagnosis/pipeline.py verify`。Guard同命令改为`guard`。执行命令改为`run`。原始来源哈希、预算、分区、梯度、输出一致性必须通过；负科学结果不阻止后续诊断。

所有新增工件仅在 `experiments/v5-sequential-diagnosis` 和 `autoresearch/loop-v5-sequential-20261018`。非阻塞获取旧全局GPU锁，有外部GPU进程即暂缓而非杀进程。阶段按任务原子保存，可恢复；最长连续运行6小时，到上限保存状态，不宣称完成。新源码独立Git提交。人工维护的程序若修复，记录源码版本，不重用不兼容工件。无无限超参搜索。

## 固定设置与新划分

目标Qwen3-8B、草稿Qwen3-0.6B，原BF16数值稳定策略/SDPA/tokenizer/greedy/non-thinking不变。L8，公共前缀位置0/64，响应前128token。目标参考仍来自冻结教师；每次草稿自回归到首个失配即可结束，因为后缀不影响贪心连续接受长度。必须记录实际proposal与首失配、检查至少L+1参考token，不能把teacher forcing准确率当接受长度。

为复用既有诊断，重新划分**开发数据**：

- adapter_train：原router_train全部480条＋原expert_train每域按固定hash取32条，共640条。
- router_fit：原expert_train中排除上面160条后，每域按独立固定hash取40条，共200条。
- checkpoint_val：原adapter_val每域固定10条，共50条。
- evaluation：原router_val全部120条。
- 前缀不跨请求分区；逐条记录ID、prompt hash、教师hash，拒绝跨分区精确重复。上述所有集合均是旧开发池，不能写成全新最终测试。原最终ID/外部集仍锁定。

原LoRA曾训练原expert_train；此事实属于共同初始化的来源，不隐瞒。新640条中480条是原适配器训练外数据。所有新方法使用相同640条目标响应，总token曝光一致。分组规则只读取adapter_train表现；router_fit不用于LoRA训练；val只选checkpoint；evaluation只用于报告不选配置。

## 第一步：固定结构，增加训练

每池5个rank16/alpha32的q_proj/v_proj LoRA，全部从同一旧MIX16起步。semantic按原任务每专家128条；random固定打乱后每专家128条；mismatch最大化旧五专家在本轮训练请求上的平均接受长度，相对每请求均值中心化，容量约束128。对并列/有唯一最佳/重分配且严格获益的样本数量单列。

另训练pooled与pooled_weighted两个通用rank16对照。weighted对旧MIX16在位置0/64首次失配token及后2token权重2，其余1，按有效权重归一化。所有专家池仍普通CE，避免把分组与token加权混为一个变化。

每专家128条、1遍、microbatch1/accumulation4、32步；pooled640条160步。lr=5e-5恒定、AdamW/weight_decay=.01、clip1、dropout .05；只监督响应前128token。专家checkpoint在2/16/32步；通用checkpoint在10/80/160步，对应总样本曝光40/320/640。早期checkpoint对应的具体样本可能不同，作为学习曲线诊断；最终各组输入与监督token总量必须完全一致。

在50请求val上记录全部checkpoint：接受长度、专家之间有差异窗口比例、完整proposal变化、相对旧MIX16变化、池Oracle/固定最佳。每池使用val前缀Oracle选择共享checkpoint，平局选更早；随后冻结。报告是否只是参数改变而行为未变。

## 第二步：先检验互补

在120请求evaluation上比较冻结池与旧MIX16/MIX80、新通用/加权通用。报告请求Oracle、前缀Oracle、最佳固定专家（仅val选择）、各专家移除损失、共同失败和唯一最佳比例。按请求聚合窗口、2000次请求bootstrap。若失配池未胜语义/随机或通用池，只能说本分组未创造足够优势；仍继续第三步测路由损失，第四步检验容量。

## 第三步：补充可区分的路由监督

对200条router_fit收集三个冻结专家池的全部接受标签。自然分布eval保持不变，不能筛选有利请求。

各池统一比较：
1. token-ID HashingVectorizer(4096,1/2gram)+Ridge(alpha10,lsqr)，复现v4轻量基线；
2. 已知prompt与前缀解码文本的字符3–5gram TF-IDF(max12000)+Ridge(alpha1,lsqr)；
3. 同2，但有专家差异的训练窗口权重5，其他1。

这里使用字符特征避免引入新下载模型；不能称深层语义编码器。所有特征只包含当前已确认内容，无真实任务标签、未来token和本轮拒绝。报告有效请求/窗口数及能区分专家的窗口数、路由选择分布、Oracle损失、同固定专家的差值。权重提高不能创造新信息。所有3种都报告，主候选预先指定char_weighted，不按evaluation选择最好结果。

## 第四步：最后比较结构

保持失配分组和640训练请求不变，在相同32步/专家与总token预算下比较：

- 基准：rank16 q/v，第一步最终32步模型（避免checkpoint选择混淆）。
- 容量：rank32 q/v，alpha64。
- 插入层：rank16 q/k/v/o/gate/up/down，alpha32。

两种新结构每种5个专家。初始化将同一旧MIX16的q/v权重拷入对应子矩阵，其余新增B矩阵置零；旧q/v缩放保持2。新增rank的A保持标准初始化。加载后检查旧矩阵逐元素相等、新B为0，并在训练输入上做初始化草稿对照，记录可能浮点差异；若初始化明显改变，则不把差值纯归因容量。

新结构仅保存最终32步，按相同50val/120eval比较Oracle、固定最佳、差异窗口比例、活跃/总适配参数、训练时间/显存。这是容量诊断，不能把更多参数说成免费收益。若Oracle改善但路由未测，该项只支持潜力，不声称实际路由加速。更改rank或插入层属于新增结构对照，不覆盖原rank16结果。

最后用evaluation每域第一条共5条，在原目标上验证基准及两新结构各一位val选出的固定专家，max_new64、L8；15次完整SD输出必须与5次AR逐token/EOS/截断一致。一次计时只作工程诊断，无正式速度或WAN主张。

## 阶段判定与交付

每阶段保留负结果、学习曲线和源码hash。keep只在预定配对改善>1e-9且Guard通过时记账；统计支持另列CI，不能把浮点舍入或任一平均增益当证据。阶段4不凭阶段2/3负结果跳过。

输出：固定manifest、旧来源hash、分组、训练/checkpoint账本、val/eval/route矩阵、四阶段报告、最终审计、autoresearch TSV/handoff以及本地证据包。四阶段完成后给出：训练不足是否缓解、互补是否增强、路由是否兑现、容量是否起作用。不中断其他任务，不自动扩展最终测试、多seed或真实WAN。

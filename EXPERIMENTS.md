# 实验说明与恢复边界

## 1.1 Motivation：接受收益是否抵得过成本

较大共享主干加多个领域 LoRA 可能同时提高草稿能力并节省完整模型副本显存。但接受率提高不必然提高最终确认输出速度：还要支付草稿生成、路由、目标验证和状态维护成本。首先测量主干规模与专门化的收益—成本曲线，再判断专家互补是否能被真实路由利用。

## 1.2 冻结数据与训练

math/GSM8K、code/Magicoder Python、general/UltraChat，固定 seed_data=20261015。每领域2000 expert_train、200 adapter_val、400 router_train、100 router_val；另有锁定的300条 id_test 仅备份，不用于本次归档评测。去重和文件 SHA 见 manifests/data.json，具体数据集固定版本见 configs/assets.local.json。

同一 Qwen3-8B 教师产生8100条记录。三个领域专家分别使用本领域2000条，两个混合适配器使用6000条；各训练2 epoch，同一样本曝光两次。混合组更新数更多，不能描述为各组总训练 FLOP 严格相等。MIX48用于与三个r16适配器的参数量预算比较，主干大小成本另报。

AdamW：lr1e-4，betas(0.9,0.999)，eps1e-8，weight_decay0.01，warmup_ratio0.05，max_grad_norm1；LoRA dropout0.05；microbatch1×accumulation16；max_seq2048；仅 assistant response 做 CE；BF16和原随机状态不改。

## 1.3 选模和需要保存的证据

每组 epoch1/2 在 adapter_val 上比较。领域专家使用对应200条，混合使用三个领域共600条。L8在公共教师前缀位置0/32/64/128/192产生草稿，计算连续匹配长度；先请求平均再领域宏平均，同分取更早 epoch。记录逐窗口proposal、接受长度、survival、样本ID、权重和源码哈希。缓存结果在GPU上重放检查后才能复用。

SNAPSHOT.json列出10组实际选中epoch和权重SHA，selected.json保留两个epoch的证据，complete.json保存训练曝光计数。仅选中epoch的权重被打包，未附优化器断点，因此这是选中模型/结果备份，而不是逐step训练续跑备份。

## 2.1 专家互补矩阵

每规模600个adapter_val请求×BASE及五个LoRA，共3600格；L8为当前范围。报告expert×source-domain均值、Oracle相对混合LoRA的差异、固定最佳专家与路由的差异及请求bootstrap区间。无有效窗口标为缺失，不能记为0。Oracle知道所有专家结果，只能作离线接受上限，不能报告在线吞吐。

## 2.2 真实路由

每规模1500条router_train/router_val轨迹，测量三个专家的公共前缀接受表现。TF-IDF word(1,2)、max_features20000、min_df2、sublinear_tf；LogisticRegression C1、balanced、seed42、max_iter1000。任务路由标签是来源领域；接受路由标签是最佳接受专家。输入只允许prompt；router_train训练，router_val验证，不把目标回答或拒绝后信息泄露给prompt路由。

## 2.3 在线投机推理

每领域随机抽8条，合计24条，seed20261017；两规模和全部方法复用相同有序请求ID。七方法：BASE、MIX16、MIX48、DOMAIN、FIXEDBEST、ROUTE-TASK、ROUTE-ACC；L8；max_new_tokens256；单次计时；每配置新进程，只驻留该配置需要的适配器，GPU计时独占。每配置10条expert_train热身不计指标，三个组件profiling请求单独保存。

AR目标模型独立进程测量：每规模24条，共48次独立AR。每规模168次SD，共336次。每请求全部token、EOS、截断状态必须与AR一致；不删除失败样本，不放宽门槛。报告有效确认token/s、端到端时延、TTFT、ms/token、接受长度、草稿/验证/路由组件时间、显存。跨GPU测得吞吐不得拼成一个硬件条件下的结果。

## 2.4 恢复与正确性

代码保留原绝对路径与来源哈希，先在新机器准备spec环境、基础模型和快照。不要向正在运行的AutoDL/4090项目覆盖文件。直接修改核心数值实现、精度、配置或路径会使旧gate失效，必须重新跑P0，不能手改哈希伪装通过。

可先在原路径运行 `python -m pytest -q scripts/continuation/test_evaluation_helpers.py` 检查编排；GPU环境需要检查驱动/实际PyTorch CUDA版本，再按 smoke_test.py 的参数运行P0。监督器已有活跃实例时禁止启动第二个。人工评测也必须使用已有GPU锁机制，不能通过绕过锁抢占计时任务。

## 2.5 本快照能支持的结论

10组seed42训练和选模已完成。矩阵/路由/在线仍按SNAPSHOT.json和逐任务complete记录判断；存在部分文件不等于该阶段已完成。旧BF16合并副本出现输出差异，保留该限制。当前24条在线范围即使完成，也不能代替完整测试、多seed、真实WAN或SLO实验。

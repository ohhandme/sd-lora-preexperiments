# V5 四阶段预实验交接：AutoDL 释放后续跑

状态：第一、二、三步完成；第四步的 10 个新结构适配器已在 AutoDL 训练完，但评估未完成，在线正确性和最终 guard 未执行。不得称四步全部完成。用户因 GPU 即将释放要求保存并交接给 4090，SSH 随后被远端关闭，停止命令未能送达。

## 已得到的结果

| 阶段 | 结果 | 当前判断 |
|---|---|---|
| 1 固定结构增加训练 | 失配池从 2→32 步，验证 Oracle 4.3265→4.6837，差异窗口 9→23 / 82 | 行为差异增加；验证有效请求 49/50 |
| 2 冻结池比较 | 120 请求、208 窗口；失配 Oracle 4.0917，语义 4.1208，随机 4.2417，MIX80 3.7667 | 失配池有互补，但未证明优于语义/随机 |
| 3 实际路由 | 失配池 86/346 个训练窗口有区分度；Hash 3.6833，字符 3.6583，加权字符 3.6583 | 更丰富特征及加权未带来收益 |
| 4 结构消融 | rank32 q/v 与 rank16 全投影层各 5 个模型训练完成 | 评估未完成，不能下结构优劣结论 |

失配池 Oracle−语义池为 −0.0292，95% 请求 bootstrap CI [−0.1667, 0.1000]。主候选（加权字符路由）相对同路由语义池 −0.0042，CI [−0.1793, 0.1708]；相对 MIX80 −0.1083，CI [−0.3750, 0.1500]。没有优势证据，也不能把跨零区间解释为证明方法相等。

640 条分组数据中 296 条旧专家平均接受长度全部平局，184 条有唯一最佳。均衡容量约束没有将任何请求分给低于其最大得分的组。当前分组依据旧专家谁更擅长，所有新专家却共同从 MIX16 起步并优化普通 CE；这尚不是按拒绝残差机制训练的专家体系。此处是机制解释假设，不是因果证明。

## 已保存与缺失

- Release `v5-sequential-handoff-20261018`：`v5-initial.tar.gz`（598,589,524 字节）、`v5-evidence-now.tar.gz`（3,062,176 字节）及校验清单。
- initial 包有 24 个完整适配器：semantic/random/mismatch 各 5 个，pooled/weighted 共 2 个，rank32 5 个，wide_0/1 共 2 个。包含各自已保存的检查点、complete.json、原始矩阵和源码。已完成工作的优化器状态没有保存，因为续跑应跳过完整训练任务。
- evidence 包更新至全部 27 个适配器训练完成之后，含 stage4_training.json。**训练账本存在不等于权重已备份。**
- wide_2、wide_3、wide_4 的权重只在远端完整 delta 包中。本机 delta 只有 39,288,832 / 220,559,614 字节，不是有效备份，未作为完整 Release 资产发布。
- 远端 delta 同时包含 core/configs、v2 脚本、冻结教师/manifest 和选中的旧专家；这些依赖也可从既有 v2 仓库与权重归档恢复。官方基础模型不在本次包中。
- 所有校验值见 `../results/v5-sequential-diagnosis/RECOVERY_MANIFEST.json`；完整矩阵在 Release 证据包中。

## 4090 接手顺序

1. 先阅读原 `experiments/v5-sequential-diagnosis/program.md`，保留其绑定与原始源码，不改写结果。下载并校验 Release 资产；解压到新目录，顺序 initial → evidence。若取回 delta，使用 initial → delta → evidence，再保留远端之后新增的最终矩阵快照。
2. AutoDL 下次恢复 SSH 后，优先取回 `/root/autodl-tmp/sd-lora-preexp/exports/v5-recovery/v5-delta.tar.gz`，验证 220,559,614 字节及 SHA256 `087b17e0f85abc3aee94f66f96dfc3f4640acea29d44f5423ec8b48ba2c673a4`。检查旧 v5 进程确已结束，禁止两个主机同时续跑同一目录。若无法取回，明确登记三个权重缺失并重新训练，不假装沿用原权重哈希。
3. 最近已验证的 4090 环境是 `/mnt/files2/dream/sd-lora-preexp/envs/spec`，Python 3.11、Torch 2.8.0+cu128、RTX4090，2026-10-17 曾通过 CUDA/BF16/SDPA 检查；基础模型当时尚未确认传完。本次不声称重新验证了 4090。确认 Qwen3-0.6B/8B 与仓库 environment/README.md 固定 revision 一致，并检查 GPU 无其他任务。
4. **原脚本不是直接可搬迁执行的程序。** common.py、v2 train.py、selected.json、assets.local.json 和 SPEC 校验含 AutoDL 绝对路径；旧 protected.json 还列有未全部打包的旧实验文件。应新建单独迁移版本，记录路径映射与变更哈希，保留原 binding、权重/数据哈希，重做迁移校验。不要直接绕过 guard，或把无法核验的旧文件记成通过。
5. 路径及依赖验证后，按原分区/预算续做 stage4：rank16 q/v 基准与 rank32 q/v、rank16 七投影层，均最终 32 步。先补齐 val/eval 接受矩阵，按请求 bootstrap；只计算未完成条目。跨 GPU 数值可能变化，先对一组已有公共前缀重放核对，若不一致应独立记录 4090 结果、重评公共基准，不能无说明混合矩阵。
6. 完成 5 条请求 × 3 结构的 15 次完整 SD/AR token、EOS、截断一致性验证，再运行迁移后的完整审计、输出 stage4.json 和最终报告。当前不存在有效 final_report.json 或 correctness.json。

原环境同路径恢复命令（只适用于路径和旧依赖完全匹配的 AutoDL，不可直接当作 4090 命令）：

```bash
cd /root/autodl-tmp/sd-lora-preexp
/root/miniconda3/envs/spec/bin/python experiments/v5-sequential-diagnosis/pipeline.py run
```

新 GPU 的剩余评估预计还需几十分钟量级，取决于基础模型准备、缺失权重取回和数值核验；不承诺固定完成时间。此实验为单 seed、复用开发数据、公共前缀离线诊断，没有最终测试、WAN 或正式加速结论。原 v1–v3 仓库文件与历史提交不修改；此次仅新增 v5 工件。

# pick_tool_token 抓取-举升 探索记录

**任务**:xArm7 + XHand 在桌面上抓起一把锤子(`textured_mesh.obj`,~19cm),RL(PPO,rl_games SAPG fork)。
动作 = CrossDex token(7 臂关节增量 + 9 eigengrasp token → RetargetNN → 12 手关节)。
**目标**:把整个锤子稳定抓起、举离桌面 20cm。

---

## 一、两个致命基础设施 bug(比任何奖励调参都重要)

### Bug 1:接触传感器 `force_matrix_w` 在多-env 下失效
- `_finger_contact_state` 用了**过滤接触** `force_matrix_w`。1 env 正常(36.7% / 35.7N),**64 env 恒 0**。
- 后果:抓握/举升奖励全部是 contact 门控的 → **之前所有多-env 训练的 grasp/lift 奖励从没真正触发**,只有 reach 在起作用。前面几十版奖励调参全是空转。
- 修复:改用**无过滤 `net_forces_w`**(净接触力,任意规模都工作)+ **逐指尖靠近门控**。commit `dbdbdc6`。
- 教训:**1-env 诊断不可信,接触传感器必须在训练规模下验证。**

### Bug 2:clearance(离桌高度)口径错误
- 用"物体**局部 AABB 包围盒**旋转到世界后的最低角"当离桌高度。包围盒比真实网格松 **~12cm**,物体一旋转,盒角凭空翘起冒出假 clearance。
- 后果:报的"举升 10cm / 18cm"**全是假的**——真实网格最低点自始至终贴在桌上(真实 clearance≈0),物体只是被**翻转竖立**。
- 修复:改用**真实网格凸包顶点**(1173 个,最低点必是其一)算真实最低点 − 真实桌面。commit `fd3de19`。
- 教训:**物体 root 高度 / 松包围盒都是可被旋转欺骗的举升代理量;必须用真实网格最低点。永远用"物体 vs 桌面"的地面真值核对"举升"说法。**

---

## 二、反复出现的 reward-hacking 行为(奖励一放松就冒出来)

| Hack | 现象 | 触发原因 | 对策 |
|---|---|---|---|
| **悬停/擦边** | 指尖贴着物体(0.7cm)不合拢 | reach occupancy 可无接触farming | palm_facing×align 方向门控;接触后关 reach |
| **Crush-launch** | 侧向挤压(pinky 285N)把物体挤射上去 | 棘轮 lift 奖励瞬时峰值 | occupancy-height 替代棘轮 + hold_quality |
| **Tip-to-vertical** | 把锤子竖起来,root 涨但整体没离桌 | lift 用 root 高度 | 改用真实 clearance |
| **手背压(back-of-hand)** | 手掌背对物体、从背面压(thumb 460N) | palm_facing 只在 reach、抓握态无约束 | palm_facing 加进 is_grasped 判据 |
| **投掷/弹飞** | 正确接近→夹击→物体弹飞→高度涨 | lift 只看高度不看手物是否同动 | valid_contact × hold_quality 双门控 |
| **压桌借力** | 用臂/手压桌面反作用力抬升 | 无 | 臂-桌几何 termination(后又移除) |
| **won't-lift** | 抓住/触到但从不往上命令臂 | 白噪声动作出不了持续上升;局部最优 | (未解决,核心难点) |

---

## 三、奖励版本时间线(本轮)

| 版本 / commit | 改动 | 结果 |
|---|---|---|
| mvp20 rerun `dbdbdc6` | 修接触sensor + 最简 mvp20 奖励(棘轮lift) | **首次真举升 15cm** — 但是 crush-launch hack |
| mvp26 关节空间 `6e6f222` | occupancy-height + 逐指尖接触门控 | 18cm — 但是 tip-to-vertical hack(假的) |
| clearance-lift `dcd02f2` | lift 改真实 clearance + 臂-桌termination | 翻转骗分被堵;倾斜从 8.5→10cm(仍偏早) |
| clean-lift gate `9b778f6` | lift × (朝向不变 × 不横漂) | 堵"乱扭凑clearance" |
| grasp-hold floor `9d5712a` | is_grasped 每步维持奖励(>hover) | 止住"退回hover";但变成**手背压** |
| palm-facing gate `41d11d4` | palm_facing+align 加进 is_grasped | 堵手背压 |
| 极简两项 `171c83a` | 砍到 reach+lift(用户要求) | 引入 fling hack + 误删 align 方向 |
| 恢复 align `e608ea0` | 方向约束加回 reach | — |
| held-gate lift `fa0f6e2` | lift × valid_contact × hold_quality | **fling 实测归零**(顶点也是0) |
| 四项阶段式 `f8e8bd8` | +R_grasp一次性 +R_success +刚体hold修正 | 修接触断崖/成功即亏/旋转误判滑移 |
| **phased 训练(5000×3)** | 上面四项式正式训 | **失败:擦边不合拢,从不举升** |

---

## 四、当前奖励(commit `f8e8bd8`,已验证结构)

```
reward = R_reach + R_grasp + R_lift + R_success

R_reach   = 2.0 × 0.5(粗核+细核) × palm_facing × align × (~is_grasped_phase)
R_grasp   = 100 × first_stable_grasp        # 一次性,stable=valid_contact & hold_quality>0.7
R_lift    = 20  × clip(clearance/0.20) × valid_contact × hold_quality
R_success = 500 × newly_successful          # 一次性(成功会结束回合)
```
- `clearance` = 真实网格凸包最低点 − 真实桌面(旋转无关)
- `align` = 0.5(1 − 指腹法线·手柄外法线),对掌压入=1,手背=0(符号已验证)
- `hold_quality` = exp(−slip_lin/0.3)·exp(−slip_ang/3.0),`slip_lin=‖v_obj−(v_palm+ω×r)‖`(**刚体补偿**,旋转持握不误判)

**已验证**:接触无断崖(连续性表)、旋转持握 hold_q=1、fling→lift=0。

---

## 五、最终训练结果(phased,5000 epoch × 3 seed):失败

| 指标 | s0 | s1 | s2 |
|---|---|---|---|
| 真实 clearance_max | 0.2mm | 0.2mm | 0mm |
| lift≥5cm / success | 0 / 0 | 0 / 0 | 0 / 0 |
| r_lift_mean | 0 | 0 | 0 |
| hold_quality_mean | 0.35 | 0.13 | 0.12 |
| is_grasped_phase | 0.08 | 0.017 | 0.0005 |

**诊断(keypoint_diag + forcelift):** 手指能贴到手柄 0.7cm、palm_facing 正确,但**只 9% 步在轻触**(thumb+mid 各 ~20N,其余指 0N),**从不合拢成握**;forcelift 显示抓握相末尾 contact=0。

**根本矛盾:** 从"指尖擦边(近、轻触)"到"合拢成力闭合握"这一步**没有专门奖励**,而 grasp bonus 又卡在"已经握稳(hold_quality>0.7)"——擦边永远到不了,桥没通。策略在 reach 局部最优就停住。

---

## 六、核心结论

1. **两个基础设施 bug(接触sensor、clearance口径)污染了绝大部分历史结果**——很多"成功/举升"是假象。修好后才第一次看清真实行为。
2. **纯奖励-shaping 反复撞同一堵墙**:擦边不握 / 握不抬 / 各种 hack。每堵一个 hack,策略就找到下一个,或退回悬停。
3. **真正没解决的是探索**:合拢一个稳固握 + 持续上举,是个长时序稀疏行为,策略从没随机撞到。
4. **下一步候选**:(A) 降 grasp bonus 门槛让"合拢"立刻变现;(B) 加"合拢/包裹"密集整形;(C) 课程 / 脚本抓握示范 bootstrap 探索。倾向 A+B,但 C(示范/课程)可能才是打穿探索墙的关键。

---

## 附:诊断工具(scripts/rl_games/)
`isgrasped_diag.py`(接触规模验证)、`keypoint_diag.py`(指尖-keypoint+palm_facing)、`forcelift_diag.py`(冻结抓握强制抬,测握能否举)、`truelift_diag.py`(凸包真实clearance vs 假包围盒)、`press_diag.py`(各body vs桌面z)、`fling_real.py`/`fling_test.py`(投掷→lift归零)、`cond_action_diag.py`(抓握后是否命令上举)。

---

## 七、探索桥接实施结果（2026-07-20）

这轮不再继续放大奖励，而是把「接近→合拢→稳定持握→持续上举」先变成可验证的控制闭环，再用示范启动探索：

- 动作扩为 21 维：`arm7 + token9 + distal residual5`。残差直接覆盖五个指尖远端屈曲关节的完整非对称可动范围，且不累积。
- 观测扩为 115 维，加入逐指接触/包裹、抓握 latch、刚体 slip、历史动作与阶段量；旧 87 维保持逐元素前缀兼容。
- grasp 判据改成方向门控的 thumb + 两个对侧指腹包裹，并用 Schmitt latch 消除接触抖动；close/contact/wrap progress 填补擦边到力闭合之间的奖励空洞。
- 加入 25 N 去预载、30 N 主动卸载、单步手目标限速与未 latch 禁止上举；持续超过 30 N 10 帧或超过 60 N 2 帧终止。
- episode horizon 从 6 s 增至 20 s；严格成功轨迹的中位完成时刻约为第 630 个控制步，短 horizon 会系统性截断成功。
- 支持从物理轨迹边界做 reverse curriculum，并同时恢复机器人关节速度、物体线/角速度和上一动作，避免中途状态注入造成虚假 slip 冲击。
- 新增脚本 oracle、状态反馈示范采集、数据合并、BC/DAgger、checkpoint 迁移与分离 actor/critic 转换工具。示范教师由可观测的上一执行动作恢复握持目标，不依赖跨步隐藏积分器。

### 严格结果

| 控制器 | 评估 | 真实 20 cm 成功 | 结论 |
|---|---:|---:|---|
| 脚本闭环 oracle | 8 env | 8/8 | 任务在当前动力学、动作空间和安全阈值下可行；最终真实 clearance 22.2–23.4 cm |
| 最佳学习 actor，seed 99 | 256 env × 1000 step | 42/256 | 49 个 env 到过 20 cm，42 个满足稳定成功 |
| 最佳学习 actor，seed 100 | 256 env × 1000 step | 41/256 | 44 个 env 到过 20 cm，41 个满足稳定成功 |
| 两 seed 合计 | 512 env | **83/512（16.2%）** | 已打穿探索墙，但远未达到稳健部署标准 |

端到端继续 BC、95% learner DAgger、option FSM 和直接 PPO fine-tune 都未超过该基线；其中最新状态反馈 DAgger actor 在 seed 99 上降到 0/256，因此已拒绝。结果说明现在的主要问题已从「从未探索到抓举」转成「全网络蒸馏/更新造成闭环分布漂移」。下一步应冻结当前 actor，只训练有界残差适配器或门控 option，并以两 seed 的严格物理成功率作为唯一晋级条件，不能再以离线 action MSE 选模。

本轮可复现交付位于忽略版本控制的
`logs/rl_games/pick_tool_token/0_bootstrap_handoff_20260720/`：包含最佳 actor、8-env oracle 数据和两组严格评估 JSON。

---

## 八、冻结基策略后的 close-window 消融（2026-07-20）

两 seed 基线的最大损失是 `touch -> grasp latch`：465 个环境碰到物体，只有 130
个形成严格 latch（28.0%）；latch 后的 `5 cm -> 20 cm -> stable success` 条件转化率已经是
83.8%、85.3%、89.2%。因此这一轮冻结最佳 actor 和 observation RMS，只允许在
「合格 pregrasp、尚未 latch」窗口做有界修正，并先在 seed 99 / 256 env / 1000 step
筛选。所有结果继续使用真实网格 clearance、严格 latch、15-step stable success 和既有力安全
终止；未过 seed 99 的候选不运行 seed 100。

| 控制器 | seed | strict | latch | latch+5 cm | 20 cm | fling* | unsafe events/env | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 冻结基线 | 99 | 42 | 71 | 58 | 49 | 10 | 41/36 | 保留 |
| 冻结基线 | 100 | 41 | 59 | 51 | 44 | 4 | 44/37 | 保留 |
| arm-zero + token-hold + distal servo | 99 | 32 | 68 | 44 | 36 | 8 | 35/31 | 拒绝 |
| **仅 close-window arm-zero** | 99 | **43** | 62 | 48 | 44 | 6 | 35/31 | 提升仅 +1，不晋级 |
| 首次 touch 后 arm-zero | 99 | 41 | 57 | 47 | 41 | 2 | 32/29 | 拒绝 |
| distal-only DAgger adapter，±0.05 | 99 | 35 | 55 | 41 | 37 | 4 | 35/29 | 拒绝 |
| distal-only DAgger adapter，±0.10 | 99 | 29 | 45 | 37 | 32 | 5 | 35/32 | 拒绝 |

手写 servo 的第一次运行还暴露了两个重要控制问题：只靠 proximity 进入会在正式 pregrasp
前闭手，使 `pregrasp_score >= 0.25` 从 154/256 降到 32/256；把 distal 当绝对命令保持则会
抹掉基策略后续有用的手部变化。加入 `pregrasp_score >= 0.25` 入口、把命令改成叠加残差并
收紧到 ±0.10 后，reach/touch 恢复，但统一同向闭合仍降低 latch。arm-zero 的小幅正结果说明
闭合时掌位稳定有价值，不过不是独立解决方案。arm-zero 与基线做同 seed 环境槽配对后是
14 个新增成功、13 个原成功丢失，属于结果重排而不是支配性提升。默认路径的两次重复回归为
42/256 和 45/256 strict；GPU PhysX 的接触 rollout 有小幅非确定性，因此晋级不能依赖单次
latch 波动或 +1 success。

监督残差采用冻结 actor latent64 + close observation suffix28 + base distal5，网络仅为
`97 -> 32 -> 5`；末层零初始化，`action[:16]` 位级不变，checkpoint 与 base SHA-256 绑定。
训练只取 95% learner-visitation DAgger 的 11,037 条 close 样本，并按 episode 隔离验证。
尽管离线 normalized MAE 达到 0.233，物理成功率仍下降；±0.10 也只能覆盖 61.4% 的教师
差值。结论是这批数据由 4 个成功、252 个失败 rollout 构成，失败/不可恢复状态的纠正标签
不适合直接蒸馏，即便只更新 distal5、严格限幅也会造成闭环分布偏移。

### 决策

1. 默认部署仍使用
   `logs/rl_games/pick_tool_token/0_bootstrap_handoff_20260720/nn/pick_tool_stage7_dagger_full_iter3_bc.pth`，严格结果保持
   83/512（16.2%）；本节所有 controller/adapter 都是显式 opt-in 诊断项，没有改变默认路径。
2. 不再继续 reward 权重、统一闭合 servo 或「失败占多数的 DAgger 数据 + action regression」。
3. 下一批数据必须先改变成功比例：在 close/micro 边界做 reverse curriculum，或在在线 rollout
   中按完整 option（不是随机单步）介入脚本教师；只保留真实完成 20 cm 且满足力安全的轨迹，再
   逐级扩大初始状态分布。
4. 学习对象应拆成 `reach actor -> close option -> lift option`。先要求 close option 从策略真实
   pregrasp 状态达到至少 80% latch，再训练 lift；最后只学习 option gate 或用带 behavior-anchor
   的 PPO 做小步联合微调。任何模型仍须先过 seed 99，再以 seed 99+100 合计 strict >=91/512、
   每 seed 相对基线下降不超过 2 个、fling <=14、unsafe env <=73 才能替换当前基线。

\* 这里旧 `fling` 字段定义为「最终未成功且曾在未 latch 时超过 5 cm」，所以会漏掉先 fling、
后恢复成功的环境槽；评估器现已另报不排除最终成功的
`ever_unlatched_clearance_ge_5cm_count`。安全 reset 后同一环境槽会开始新尝试，而漏斗继续累计，
因此表中是 1000-step campaign 的 ever-event，不是严格逐 episode 条件概率。

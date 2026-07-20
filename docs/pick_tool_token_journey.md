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

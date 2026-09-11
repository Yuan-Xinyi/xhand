# MANUS 手套 → XHand 遥操作

把 Manus Meta Gloves Pro 接到真实 XHand 上的完整记录。写给下一个接手的人 —— 包括为什么是这个架构、以及所有踩过的坑。

---

## 1. 为什么要做这个

repose-cube 策略在仿真里成功率 96%,真机上却卡死:cube 一旦和掌面形成平面接触就再也转不动。仿真消融已经排除了 cube 尺寸、质量、穿透、起始位姿、逐连杆摩擦 —— 都不是主因。

剩下两种可能:

- **硬件做不到** —— XHand 的运动学/接触条件本身就转不动这个 cube
- **闭环做不到** —— 硬件能做到,但感知-动作回路不行

**人来遥操作就能把这两者分开**:如果人操纵机械手能转动 cube,那硬件没问题,该改的是策略/闭环;如果人也转不动,那该改的是指面材质、cube 表面、或者换任务。

这就是这套管线存在的唯一理由。

---

## 2. 授权:为什么非要用 Unity

Manus 的 license 是按功能组件卖的,绑在硬件 dongle 上。实测这套设备的激活情况:

| 组件 | 状态 |
|---|---|
| recording / exporting / advanced exporting | ✅ |
| **Unity plugin** / Unreal plugin / MotionBuilder plugin | ✅ |
| **sdk** | ❌ |
| **integrated** | ❌ |

关键后果:`libManusSDK.so` 和 `libManusSDK_Integrated.so` **两个都用不了**。

- `integrated` 模式(dongle 直插 Linux,不需要 Manus Core)—— 实测能跑到 110 Hz 出数据,然后被 license 挡掉
- `sdk` 模式(Linux SDK 连 Windows 上的 Manus Core)—— 同样被挡
- ROS2 包用的是同一个 `.so`,一样被挡

**Manus Core 只支持 Windows 10/11**,Linux 上跑不了。所以拓扑被授权钉死:

```
dongle 必须在 Windows 上  →  Manus Core  →  Unity 插件  →  自写桥接  →  UDP  →  Linux
```

Unreal 插件要 40 GB+,MotionBuilder 要 $2300/年,所以 **Unity 是唯一免费且现实的门**。

> 曾经认真评估过"录制 + 导出 CSV"这条路(recording/exporting 是激活的)。技术上可行,但录制期间数据不出 Manus Core,要按停止再点导出 —— 是一次一个 take,take 内部无法纠偏。3.1.0+ 有 watch folder 自动导出 + UDP 远程触发(`<CaptureStart>` XML,Qualisys 档位端口 8989,实测可用),理论上能做成 3~6 秒一轮的分块闭环。但 Unity 实时流打通后就不需要了。

---

## 3. 架构

两条并行的数据通路,**都可以同时开**(端口不同):

```
                      Windows                           Linux
                 ┌──────────────┐
   手套 ─dongle─▶│  Manus Core  │
                 └──────┬───────┘
                        │
                 ┌──────▼───────────┐
                 │  Unity 编辑器     │
                 │  (Play 模式)      │
                 │                  │
                 │ ManusUdpBridge ──┼── udp:9881 ──▶ manus_node.py ──┐
                 │  (20 通道角度)    │      JSON                      │
                 │                  │                                 │ udp:51234
                 │ ManusSkeleton    │                                 │ (12 关节,
                 │   Bridge ────────┼── udp:9882 ──▶ manus_skel_node ─┤  二进制)
                 │  (3D 骨架树)      │      JSON                      │
                 └──────────────────┘                                 │
                                                                      ▼
                                        real_node.py ◀────────────────┘
                                        (RS-485, /dev/ttyUSB0 @3M)
                                              │
                                              ▼
                                          真实 XHand
```

关键设计:**两个 manus 节点都输出 `protocol.py` 定义的同一个线协议**,所以它们是 `perception_node.py`(相机/WiLoR)的**平行替代品**。`real_node.py` 和 `teleop_sim.py` 完全不知道上游换了什么,一行都不用改。

### 文件

| 文件 | 位置 | 作用 |
|---|---|---|
| `ManusUdpBridge.cs` | Unity `Assets/` | 发 20 个 ergonomics 角度 |
| `ManusSkeletonBridge.cs` | Unity `Assets/` | 发 3D 骨架树(局部位姿 + 父子关系) |
| `manus_node.py` | `tools/teleop/` | 20 通道 → 12 关节 |
| `manus_skel_node.py` | `tools/teleop/` | 骨架 → FK → MANO 21 点 → DexPilot → 12 关节 |
| `manus_teach.py` | `tools/teleop/` | **示教标定**(见下) |
| `manus_calibrate.py` | `tools/teleop/` | 量程标定(早期方法,已被 teach 取代) |
| `manus_csv_to_xhand.py` | `RealExperiments/manus_bridge/` | 离线 CSV 重定向 + 共享映射表 |
| `manus_trigger_probe.py` / `_sweep.py` | scratchpad | Manus Core 录制远程触发探针 |

---

## 4. 映射问题:失败五次的那件事

这是整个项目最费时间的部分,也是最值得记录的教训。

### 问题

手套给 20 个数(5 指 × 张开/掌指/近指/远指),XHand 有 12 个关节。要建立 20 → 12 的映射。

### 我试过的方法,和它们怎么失败的

**方法一:按通道名字接线。** `ThumbMCPSpread` → 拇指侧摆关节,`ThumbMCPStretch` → 拇指弯曲关节,以此类推。量程用解剖学名义值。

失败。手指弯不满(名义量程比实际宽 5 倍),拇指方向反。

**方法二:按 URDF 几何推方向。** 算出每个关节轴在掌坐标系里的朝向,推断正方向对应什么动作,据此决定要不要取反。

失败,而且错得隐蔽。**我把掌坐标系的轴认错了** —— 四指根部都在 `z≈0.105`、`y≈-0.006`、沿 `x` 排开,说明 **+z 是手指伸出方向、掌法线是 y**,而我一直当成 z 是掌法线。基于错轴推出的每个结论都是废的。

**方法三:实测量程 + 取反。** 写了 `manus_calibrate.py`,让操作者摆 7 个姿势,实测每个通道的真实行程。量程问题解决了(dipstretch 实际只有 10–37°,不是 80°),但方向还是错。

**方法四:换通道。** 从标定数据发现拇指两个通道的语义和名字对不上:

| 通道 | 拇指张开 | 拇指横过掌心 | 拇指抬起 | 实际测的是 |
|---|---|---|---|---|
| `mcpspread` | −7.3 | +12.5 | **+36.0** | 掌面外展(抬起) |
| `mcpstretch` | **+43.1** | **+19.8** | +48.7 | 平面内对掌 |

于是把两个通道对调。还是不对。

**方法五:3D 骨架 + DexPilot。** 改成让 Unity 发 3D 关节点,喂给相机路径已经在用的 DexPilot 重定向器,让几何决定一切。

架构上是对的(而且有个额外好处:和 RL 策略蒸馏用的是同一套重定向),但实际跑出来仍然不对 —— 大概率是我建的手掌规范坐标系(`canonical_frame` 里那三个基向量)和 MANO 的约定不一致。这条路没走完。

### 为什么这四次都注定失败

**它们都在猜 MANUS 那一侧的语义** —— 通道归属(20 选 1 × 12)、符号(2¹² 种)、量程。三者互相纠缠,而且**符号错和归属错在真机上长得一模一样**,所以每次只能改一个变量再上真机试一遍。

这不是设计,是穷举搜索。而且验证一轮要几分钟,人还得站在机器人前面。

### 最终解法:反过来做(`manus_teach.py`)

**把标定方向倒过来:不去猜人手,而是让机械手当老师。**

```
我把机械手摆到一个关节向量 q      ← 我自己下发的指令,标签精确已知
        ↓
你看着它,把自己的手摆成一样      ← 对应关系由人的眼睛建立
        ↓
我记录你的 20 维特征 e
        ↓
重复 19 次 → 岭回归拟合 e → q
```

**归属、符号、量程、交叉耦合,全部从回归里解出来。我一个都不用猜。**

唯一需要的判断是"我的手看起来像不像那只机械手",而这是人做得最可靠、我做得最不可靠的事。

### 效果

首次实测(19 个姿势,2.5 分钟):

```
          joint     R^2   主导通道
   index_joint1   0.928   index.mcp, index.pip
   index_joint2   0.972   index.pip, index.mcp
  middle_joint0   0.954   middle.pip, middle.mcp
  middle_joint1   0.985   middle.pip
    ring_joint0   0.950   ring.pip
    ring_joint1   0.981   ring.pip
   pinky_joint0   0.907   pinky.pip, pinky.mcp
   pinky_joint1   0.966   pinky.pip
   thumb_joint0   0.758   thumb.spread, thumb.pip
   thumb_joint2   0.861   thumb.pip, thumb.dip
   thumb_joint1   0.489   ← 拟合差
   index_joint0   0.298   ← 拟合差
```

**最强的旁证:四指每个关节都被它自己手指的特征主导。** 这个结构是回归自己找出来的,我没有告诉它任何对应关系。前四次手写映射从来没达到过这个水平。

两个弱项的解释:

- `index_joint0` 是食指侧摆,总行程只有 ±10°,机械手上肉眼几乎看不出来,人没法可靠复现。对转 cube 影响很小。
- `thumb_joint1` 是拇指那个一直有问题的自由度,人摆的时候容易和其它拇指自由度耦合在一起。

### 一个重要性质

**低 R² 退化成"这个关节不怎么动",而不是"反着动"。**

回归不会凭空产生符号翻转 —— 数据说什么方向就是什么方向,只是响应弱。这和手写映射的失效模式完全不同(那个一错就整个反向,直接把物体顶飞)。

所以拟合差是**安全的失败**,这是这个方法除了准确率之外的另一个好处。

### 后续改进(已实现)

1. **岭回归前标准化特征。** 首次实测 alpha 落在搜索范围上界 1000 —— 这是"惩罚在和量纲打架而不是和噪声打架"的信号。ergonomics 各通道量纲差很多(10° 到 130°),ridge 一视同仁地惩罚系数,等于对不同特征施加了完全不同强度的正则。标准化后 alpha 降到 0.1(落在范围内部)。A/b 会折回原始尺度,所以存下来的映射仍然吃原始特征。
2. **存原始采样。** 首次会话只存了拟合结果,改进估计器就得让人重摆 19 次。现在 `E`/`Q` 一起存进 json,`--refit` 可以离线重算,不碰硬件也不碰手套。

---

## 5. 踩过的坑

### Unity / Manus 插件

**`CommunicationHub` 不是 MonoBehaviour,不要往场景里拖。** 它自己的文档注释写着 *"This component should not be added to the scene manually."* 插件里根本没有它的预制体。`ManusManager` 带 `[InitializeOnLoad]`,编辑器一加载就自动把通信跑起来。场景里只需要放你自己的桥接脚本。

**有两个同名的 `ErgonomicsStream`。** `CoreSDK.ErgonomicsStream` 是扁平的 marshalling 结构体(定长 `ErgonomicsData[32]` + `dataCount`),而 `CommunicationHub.ergonomicsData` 返回的是插件自己包装的 `CommunicationHub.ErgonomicsStream`(`List<CoreSDK.ErgonomicsData>`,没有 `dataCount`)。用错会编译不过。只有内层的 `ErgonomicsData`(`isUserID` + `float[40]`,offset 20 = 右手)是 SDK 原生类型。

**原始骨架是局部位姿树,不是世界坐标。** `--dump` 出来每个非掌骨节点都是 `(0, 0, 骨节长度)`,x=y=0。必须连同**旋转和 `parentId`** 一起取,在接收端做正运动学才能得到真实 3D 位置。判据:合成后 `wrist→中指尖 ≈ 0.18–0.21 m`、`食指根→小指根 ≈ 0.06–0.08 m`。

**Unity 失焦会把 `Update()` 降到极低频。** 你切到 Linux 终端看数据的那一刻流就停了,看起来像断联。`Application.runInBackground = true;` 解决。

**改任何 `Assets/` 下的文件都会触发重编译并自动退出 Play。** 改完脚本记得重新按 ▶。

**在 Play 模式下创建的 GameObject,退出 Play 时会被销毁。** 建好组件记得 **Ctrl+S 存场景**。

**Unity 6 把 `Create > C# Script` 挪到了 `Create > Scripting > MonoBehaviour Script`。** 更省事的做法:直接用记事本存一个 `.cs` 到 `Assets/`,Unity 会自动认(注意"保存类型"选**所有文件**,否则会存成 `.txt`)。

**Unity 安装本身的坑:** 2022.3 LTS 过了支持期,免费 Personal 授权装不了(需要 Industry/Enterprise),所以只能用 Unity 6(6000.0.x LTS 最稳)。装之前**先加杀毒排除项** —— `PackageManager\Server\UnityPackageManager.exe` 和 `ProjectTemplates` 被 Defender 隔离过,症状是建项目时报 `com.unity.template.3d not found`。模块一个都不用勾(不需要 Visual Studio,Unity 自带 C# 编译器;不需要任何平台 Build Support)。

**手套会自动休眠。** 休眠后 `ergonomicsData` 是空的,桥接脚本一声不吭地 return,Console 什么都不打印,和"没在 Play"长得一模一样。现在两个桥接都会每秒打印一行说明为什么不发。

### 网络

**UDP 满缓冲丢新留旧。** Linux 的 UDP socket 缓冲满了之后丢的是**新**包、留的是旧包。倒计时/等待期间不读 socket 会导致缓冲塞满,之后拿到的全是陈旧数据,甚至整段采样为空。**所有等待循环里都要持续排空 socket。** 这个坑在这个项目里出现过至少三次(标定工具、teach 工具、早期 FoundationPose 管线)。

**DHCP 换 IP 会静默切断整条链路。** Linux 的 wifi 地址从 `192.168.1.33` 变成 `.36`,Unity 还在往 `.33` 发,包进了虚空。而且**每一层从自己的角度看都是健康的**:Manus Core 在推数据、Unity 打印 `hub=up`、Linux 礼貌地问"Unity 在 Play 吗" —— 这种全绿的故障最费时间。

解法:**发子网广播地址 `192.168.1.255`**(socket 要 `EnableBroadcast = true`)。接收端本来就 bind `0.0.0.0`,所以操作者机器的 IP 不再是关键路径上的东西。

**双网卡同网段。** 这台 Linux 有 `eno3 192.168.1.10`(有线)和 `wlo1 192.168.1.36`(无线),Windows 只在无线那一侧。Linux → Windows 要手动加路由:

```bash
sudo ip route replace 192.168.1.17/32 dev wlo1 src 192.168.1.36
```

(非持久,重启/换 IP 后失效。这条只影响 Linux 主动发起的方向 —— 手套数据是单向 Windows→Linux,不需要它。)

### 真机驱动

**`--ping` 不通不代表手不在。** 固件对 `0x13` 版本查询不回包,但 `0x02` 正常。用 `real_node.py --read`(只读,零运动)确认手是否上电。

**Ctrl+C 按一次就够。** 连按会打断串口的 `close()`,留下 traceback 和偶尔的端口忙。级联关闭要几秒,等它。

---

## 6. 怎么跑

### 一次性准备

Windows:
1. Manus Core 运行,手套连上并校准好
2. Unity 打开项目,Play 模式
3. 场景里有 `ManusUdpBridge`(9881)或 `ManusSkeletonBridge`(9882),`Linux Host` = `192.168.1.255`

Linux:
```bash
cd /disk2/xhand_inhand/xhand_inhand/.claude/worktrees/inspiring-grothendieck-102c72/tools/teleop
conda activate wilor
```

### 确认数据在流

```bash
python manus_node.py --listen-only --print          # ergonomics 流
python manus_skel_node.py --dump                    # 骨架流(打印节点清单 + FK 判据)
```

### 标定(每个操作者做一次,2.5 分钟)

⚠️ 机械手会动,先把 cube 和障碍物拿开。

```bash
python manus_teach.py --out manus_fit.json
```

19 个姿势。**你只需要看着机械手,把自己的手摆成一样。** 两条流哪条有数据就用哪条。

结束时打印每关节 R²。全部 > 0.6 算合格;低的那几个它会标出来。

想改拟合参数不用重摆:

```bash
python manus_teach.py --refit manus_fit.json --out manus_fit2.json
```

### 空跑验证方向

```bash
python manus_node.py --fit manus_fit.json --listen-only --print
```

### 上真机

```bash
python real_node.py --read      # 只读,确认手上电
python real_node.py --open      # 张开到已知位姿

python teleop.py --manus --real \
  --real-args "--max-speed 1.0 --tor-max 150" \
  --manus-args "--fit manus_fit.json"
```

跟随稳定后再把 `--max-speed` 提到 2.5、`--tor-max` 提到 300。

安全层始终生效:URDF 限位夹紧、速度限幅、EMA 平滑、首个目标取实测手位(不会跳)、2 秒断流看门狗(保持位姿,不抽搐)。

---

## 7. 遗留

- **`thumb_joint1`(R² 0.489)和 `index_joint0`(0.298)拟合差。** 重跑一次标定(现在有特征标准化了)大概率能改善。示教时注意只动机械手正在动的那一节,别连带整个拇指。
- **标准化修复后还没重新标定过。** 现用的 `manus_fit.json` 是修复前那次的。
- **3D 骨架 + DexPilot 那条路(`manus_skel_node.py`)没走完。** FK 和节点识别都验证通过了,卡在 `canonical_frame` 的基向量约定和 MANO 不一致。如果要捡起来,症状对应关系:整只手转了 90° / 食指小指互换(镜像)/ 手指朝手背弯 —— 各对应改一个基向量,不需要再逐关节试符号。
- **`real_node.py` 没有 npz 记录。** 遥操作跑完没有数据可分析,只能靠肉眼描述。加上之后能直接读"手指有没有跟上、哪些关节贴限位、cube 转动时的接触模式"。
- **正事还没做:** 这套管线是为了回答"XHand 到底能不能把 6.2cm 的 cube 转起来"。管线通了,实验本身还没跑。

---

## 8. 一句话总结

**不要按名字接线,不要在纸上推几何,让机械手当老师。**

标定的方向决定了错误的代价:猜人手的语义,错了只能上真机试,一轮几分钟且不知道错在归属还是符号;让机械手示教,标签精确已知,对应关系由人眼建立,剩下的交给回归 —— 而且拟合差只会表现为"跟得懒",不会变成"反着动"。

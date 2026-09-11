# Manus 手套数据:Unity(Windows)→ Ubuntu 实时链路

把 Manus Meta Gloves Pro 的手指数据实时送到 Ubuntu 上的完整方案。**含全部源码,照着做可复现。**

---

## 1. 为什么必须绕道 Unity

通常的做法是直接用 Manus 官方 C++ SDK,但本套设备做不到。

Manus 的授权按功能组件出售,绑定硬件 dongle。实测激活情况:

| 组件 | 状态 |
|---|---|
| recording / exporting / advanced exporting | ✅ |
| **Unity plugin** / Unreal plugin / MotionBuilder plugin | ✅ |
| **sdk**(C++ SDK 远程模式) | ❌ |
| **integrated**(C++ SDK 直连 dongle) | ❌ |

后果:

- `libManusSDK.so`(远程模式,Linux SDK 连 Windows 上的 Manus Core)—— 被拒
- `libManusSDK_Integrated.so`(dongle 直插 Linux,不需要 Manus Core)—— 实测能取到 110 Hz 数据**然后被 license 拒绝**
- SDK 自带的 ROS2 包用的是同一个动态库,一样不可用

加上 **Manus Core 只支持 Windows 10/11**(官方明确不支持 Linux),Unreal 插件要 40 GB+,MotionBuilder 是付费软件 —— **Unity 插件是唯一免费可行的数据出口**。

于是方案变成:在 Unity 里写一个 C# 组件读插件的数据,用 UDP 发给 Ubuntu。

> 先确认自己的授权状态再决定要不要照抄本方案:如果你的 license 激活了 `sdk` 或 `integrated`,直接用官方 C++ SDK 更简单。在 Manus Core 的 license 页面可以看到组件列表。

---

## 2. 环境版本

### Windows 侧

| 项 | 版本 |
|---|---|
| 操作系统 | Windows 10/11 64-bit |
| Manus Core | 3.0.x |
| Manus Unity 插件 | ManusUnityPlugin **v3.1.1** |
| Unity 编辑器 | **6000.0.83f1 (LTS)** |

> Unity 版本说明:Manus 插件官方支持 Unity 6 / 2022.3 / 2021.3 / 2020.3。但 2022.3 已过支持期,**免费 Personal 授权装不了**(需 Industry/Enterprise),所以实际只能用 Unity 6,选 `6000.0.x LTS` 最稳。

### Ubuntu 侧

| 项 | 版本 |
|---|---|
| 操作系统 | **Ubuntu 22.04.5 LTS**,内核 5.15.0-174-generic |
| Python | 3.10(接收端只用标准库,3.8+ 都行) |

### 网络

两台机器在同一个局域网。本文示例中 Windows 是 `192.168.1.17`,Ubuntu 是 `192.168.1.x`(**会变,见 §5.3**),子网广播地址 `192.168.1.255`。

---

## 3. 架构

```
   手套 ──无线 dongle──▶ ┌──────────────┐
                        │  Manus Core  │   Windows
                        └──────┬───────┘
                               │ 进程内
                        ┌──────▼──────────────┐
                        │  Unity 编辑器        │
                        │  Manus 插件 (C#)     │
                        │        ↓            │
                        │  ManusUdpBridge.cs  │  ← 我们写的,§4
                        └──────┬──────────────┘
                               │  JSON / UDP :9881
                               │  发往子网广播 192.168.1.255
                               │  ~90 Hz
                               ▼
                        ┌─────────────────────┐
                        │  Ubuntu 22.04       │
                        │  Python 接收端       │  ← §5
                        └─────────────────────┘
```

数据从手套到 Ubuntu 一共三跳,只有中间那跳需要自己写。

---

## 4. Unity 侧(Windows)

### 4.1 装 Unity

1. 装 Unity Hub,登录账号,**激活免费 Personal 许可证**(齿轮 → Licenses → Add → Get a free personal license)。不激活的话 Install Editor 按钮不出现。
2. Hub → Settings → Installs → **Installs location 改到非 C 盘**
3. ⚠️ **装之前先加杀毒排除项**(管理员 PowerShell):

```powershell
foreach ($p in @("D:\Unity", "$env:LOCALAPPDATA\Unity", "$env:APPDATA\UnityHub")) {
    New-Item -ItemType Directory -Path $p -Force | Out-Null
    Add-MpPreference -ExclusionPath $p
}
```

4. Install Editor → **Unity 6000.0.x (LTS)** → **模块一个都不勾**(不需要 Visual Studio,Unity 自带 C# 编译器;不需要任何平台 Build Support;Documentation 尤其不要)
5. 装完**先验证完整性**再建项目:

```powershell
Get-ChildItem (@("$env:ProgramFiles\Unity\Hub\Editor") + (Get-PSDrive -PSProvider FileSystem | % { $_.Name + ':\Unity' }) | ? { Test-Path $_ }) -Directory -EA 0 | % {
    $p = $_.FullName + '\Editor\Data\Resources\PackageManager'
    "$($_.Name)  Server=$(Test-Path ($p + '\Server\UnityPackageManager.exe'))  Templates=$(Test-Path ($p + '\ProjectTemplates'))"
}
```

`Server` 和 `Templates` 都必须是 `True`。任何一个 `False` 说明文件被杀毒隔离了,重装。(这一步不验的话,后面建项目会报 `com.unity.template.3d not found`,很难联想到是杀毒干的。)

### 4.2 建项目并导入 Manus 插件

1. New Project → **3D (Built-In Render Pipeline)** → 位置选非 C 盘
2. 菜单 `Assets` → `Import Package` → `Custom Package` → 选 `ManusUnityPlugin_v3.1.1.unitypackage` → `Import`
3. 等编译完,Console 不该有红色错误

**注意:插件不需要往场景里拖任何东西。** `CommunicationHub` 不是 MonoBehaviour(它自己的注释写着 *"This component should not be added to the scene manually"*),`ManusManager` 带 `[InitializeOnLoad]`,编辑器一加载就自动把通信跑起来了。

### 4.3 桥接脚本

在 `Assets/` 下新建 `ManusUdpBridge.cs`(文件名必须和类名一致):

> Unity 6 把 `Create > C# Script` 挪到了 `Create > Scripting > MonoBehaviour Script`。更省事的做法是用记事本直接存一个 `.cs` 到 `Assets/` 目录,Unity 会自动认 —— 注意"保存类型"要选**所有文件**,否则会存成 `.txt`。

```csharp
// Forwards MANUS glove ergonomics to a listener on Linux over UDP.
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Manus;          // CommunicationHub / ManusManager

public class ManusUdpBridge : MonoBehaviour
{
    [Tooltip("Where the Linux listener is. Prefer the subnet broadcast address " +
             "(x.y.z.255) so a DHCP lease change on the Linux box cannot silently " +
             "strand the stream -- the listener binds 0.0.0.0 and takes it either way.")]
    public string linuxHost = "192.168.1.255";
    public int linuxPort = 9881;
    [Tooltip("Uncheck to stream the left glove instead")]
    public bool rightHand = true;
    [Tooltip("Send at most this many packets per second")]
    public float sendRateHz = 90f;
    [Tooltip("Log progress to the Console so you can verify movement")]
    public bool debugLog = true;

    UdpClient _udp;
    IPEndPoint _dst;
    float _next;
    float _nextReport;
    int _sent;
    readonly StringBuilder _sb = new StringBuilder(512);

    void Start()
    {
        // Without this the editor throttles Update() to a crawl the moment its
        // window loses focus -- which is exactly when the operator looks at the
        // robot -- and the stream looks like it disconnected.
        Application.runInBackground = true;

        // Touch the manager so the hub is definitely constructed and running
        // before we start reading its static stream.
        var hub = ManusManager.communicationHub;
        _udp = new UdpClient();
        _udp.EnableBroadcast = true;   // required if linuxHost is a .255 address
        _dst = new IPEndPoint(IPAddress.Parse(linuxHost), linuxPort);
        Debug.Log($"[ManusUdpBridge] hub={(hub != null ? "up" : "NULL")}, sending " +
                  $"{(rightHand ? "RIGHT" : "LEFT")} hand to {linuxHost}:{linuxPort}");
    }

    void Update()
    {
        if (Time.unscaledTime < _next) return;
        _next = Time.unscaledTime + 1f / Mathf.Max(1f, sendRateHz);

        // NOTE: this is CommunicationHub.ErgonomicsStream, the plugin's own
        // wrapper -- a List<CoreSDK.ErgonomicsData>, NOT the flat SDK struct
        // with its fixed array and dataCount.  Only the inner ErgonomicsData
        // (isUserID + float[40]) comes straight from the SDK.
        var stream = CommunicationHub.ergonomicsData;
        if (stream.data == null || stream.data.Count == 0)
        {
            // Silence here is ambiguous -- asleep gloves, a dropped Manus Core
            // link and "not in Play mode" all look identical from Linux. Say so.
            Report("no ergonomics data (gloves asleep or Manus Core not streaming)");
            return;
        }

        int offset = rightHand ? 20 : 0;
        foreach (var ergo in stream.data)
        {
            if (ergo.isUserID || ergo.data == null || ergo.data.Length < offset + 20) continue;

            bool any = false;
            for (int i = 0; i < 20; i++)
                if (Mathf.Abs(ergo.data[offset + i]) > 1e-4f) { any = true; break; }
            if (!any)
            {
                // All-zero for the hand we asked for. Usually means the glove
                // being worn is the other one -- flip "Right Hand" to check.
                Report($"{stream.data.Count} glove(s) streaming but all 20 " +
                       $"{(rightHand ? "RIGHT" : "LEFT")} channels are zero");
                continue;
            }

            _sb.Clear();
            _sb.Append("{\"ergo\":[");
            for (int i = 0; i < 20; i++)
            {
                if (i > 0) _sb.Append(',');
                _sb.Append(ergo.data[offset + i].ToString("F3", CultureInfo.InvariantCulture));
            }
            _sb.Append("]}");

            byte[] bytes = Encoding.ASCII.GetBytes(_sb.ToString());
            _udp.Send(bytes, bytes.Length, _dst);
            _sent++;
            if (debugLog && _sent % 90 == 0)
                Debug.Log($"[ManusUdpBridge] {_sent} packets, thumbStretch={ergo.data[offset + 1]:F1} " +
                          $"indexStretch={ergo.data[offset + 5]:F1}");
            return;   // one glove per frame is enough
        }
    }

    // Throttled so a persistent fault states itself once a second instead of
    // drowning the Console at frame rate.
    void Report(string why)
    {
        if (!debugLog || Time.unscaledTime < _nextReport) return;
        _nextReport = Time.unscaledTime + 1f;
        Debug.LogWarning($"[ManusUdpBridge] not sending: {why}");
    }

    void OnDestroy()
    {
        _udp?.Close();
        Debug.Log($"[ManusUdpBridge] stopped after {_sent} packets");
    }
}
```

### 4.4 挂到场景

1. Hierarchy 空白处右键 → `Create Empty`
2. 选中它 → Inspector → `Add Component` → 搜 `Manus Udp Bridge`(搜不到就把脚本从 Project 窗口**直接拖**到 Inspector 上)
3. 确认 `Linux Host` = `192.168.1.255`、`Linux Port` = `9881`
4. **Ctrl+S 保存场景** ← 别漏。Play 模式下建的物体退出 Play 时会被销毁;场景不存,重开 Unity 也会没。
5. 按 ▶

Console 应立即出现:

```
[ManusUdpBridge] hub=up, sending RIGHT hand to 192.168.1.255:9881
```

### 4.5 三处非显然的实现细节

**① `Application.runInBackground = true`**
不加的话,Unity 编辑器窗口一失去焦点就把 `Update()` 降到极低频。你切到 Ubuntu 终端看数据的那一刻流就停了,看起来像断联。

**② 两个同名的 `ErgonomicsStream`**
`CoreSDK.ErgonomicsStream` 是扁平的 marshalling 结构体(定长 `ErgonomicsData[32]` + `dataCount`);而 `CommunicationHub.ergonomicsData` 返回的是插件自己包装的 `CommunicationHub.ErgonomicsStream`(`List<CoreSDK.ErgonomicsData>`,**没有 `dataCount`**)。用错会编译不过(`.Count` vs `.Length`)。只有内层的 `ErgonomicsData` 是 SDK 原生类型。

**③ 右手数据在 offset 20**
`ErgonomicsData.data` 是 `float[40]`:前 20 个是左手,后 20 个是右手。

---

## 5. Ubuntu 侧

### 5.1 数据格式

UDP 载荷是一行 ASCII JSON,约 90 Hz:

```json
{"ergo": [20 个浮点数]}
```

**单位是度。** 顺序固定为 5 指 × 4 通道,拇指在前:

| 下标 | 含义 |
|---|---|
| 0–3 | thumb: MCPSpread, MCPStretch, PIPStretch, DIPStretch |
| 4–7 | index: 同上 |
| 8–11 | middle: 同上 |
| 12–15 | ring: 同上 |
| 16–19 | pinky: 同上 |

`MCPSpread` = 手指张开(侧摆),`MCPStretch` / `PIPStretch` / `DIPStretch` = 三个指节的弯曲。

> ⚠️ 通道的**名字和它实际测量的东西不一定一致**,尤其是拇指。实测:`thumb.MCPSpread` 在拇指**抬离掌面**时最大(不是张开),`thumb.MCPStretch` 才是掌平面内的对掌动作。如果要把这些数映射到别的机械手上,**务必实测,不要照名字理解**。

### 5.2 接收端(完整可运行)

只用标准库,不依赖任何第三方包:

```python
#!/usr/bin/env python3
"""Minimal receiver for the MANUS glove stream from Unity. Standard library only."""
import argparse
import json
import socket
import time

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHANNELS = ("MCPSpread", "MCPStretch", "PIPStretch", "DIPStretch")
LABELS = [f"{f}.{c}" for f in FINGERS for c in CHANNELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0",
                    help="0.0.0.0 receives both unicast and subnet broadcast")
    ap.add_argument("--port", type=int, default=9881)
    ap.add_argument("--raw", action="store_true", help="print all 20 channels")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.settimeout(1.0)
    print(f"listening on udp://{args.bind}:{args.port}", flush=True)

    n, t0 = 0, time.time()
    while True:
        try:
            payload, addr = sock.recvfrom(4096)
        except socket.timeout:
            print("  no packets (is Unity in Play mode?)", flush=True)
            continue
        try:
            ergo = json.loads(payload.decode("ascii"))["ergo"]
        except (ValueError, KeyError, UnicodeDecodeError):
            continue
        n += 1

        now = time.time()
        if now - t0 >= 1.0:
            print(f"\n{n} pkt/s from {addr[0]}", flush=True)
            if args.raw:
                for i in range(0, 20, 4):
                    print("   " + "  ".join(f"{LABELS[i + k]:>16s}={ergo[i + k]:7.2f}"
                                            for k in range(4)), flush=True)
            else:
                print("   " + "  ".join(f"{f}={ergo[4 * i + 2]:6.1f}deg"
                                        for i, f in enumerate(FINGERS)), flush=True)
            n, t0 = 0, now


if __name__ == "__main__":
    main()
```

跑:

```bash
python3 manus_recv_example.py --raw
```

正常输出:

```
listening on udp://0.0.0.0:9881

90 pkt/s from 192.168.1.17
    thumb.MCPSpread=   9.22  thumb.MCPStretch=  65.29  thumb.PIPStretch=  84.49  thumb.DIPStretch=  46.09
    index.MCPSpread=   1.10  index.MCPStretch=  54.13  index.PIPStretch= 122.90  index.DIPStretch=  11.50
    ...
```

### 5.3 为什么发广播而不是单播

**这是本方案最重要的一个设计决定。**

最初是单播到 Ubuntu 的固定 IP。某次 Ubuntu 的 WiFi DHCP 租约变更,地址从 `192.168.1.33` 变成 `.36`,Unity 仍然往 `.33` 发 —— 包进了虚空。

麻烦的是**每一层从自己的角度看都是健康的**:

- Manus Core:手套在线,数据在推
- Unity Console:`hub=up`,还在打印发包计数
- Ubuntu:"没收到包,Unity 在 Play 模式吗?"

三边都"正常",排查花了很久。

改发子网广播 `192.168.1.255` 后,接收端 bind `0.0.0.0` 就能收到,**操作者机器的 IP 不再位于关键路径上**。代价只是同网段其它机器会收到这些包(局域网内无所谓)。

C# 侧必须设 `_udp.EnableBroadcast = true`,否则发 `.255` 会抛异常。

### 5.4 UDP 缓冲区的坑

如果接收端有"等待/倒计时"之类不读 socket 的阶段,**必须在等待期间持续排空 socket**:

```python
def drain(sock):
    while True:
        try:
            sock.recv(65535)
        except (BlockingIOError, OSError):
            return
```

原因:Linux 的 UDP socket 缓冲区满了之后,内核丢弃的是**新到的包**、保留的是旧包。几秒不读就会塞满,之后拿到的全是陈旧数据,甚至整段采集为空。这个坑在本项目里踩了三次。

遥操作只关心最新一帧,所以正确的读法是"排空到最后一个包,只用它":

```python
newest = None
while True:
    try:
        newest = sock.recv(65535)
    except (BlockingIOError, OSError):
        break
# 用 newest
```

---

## 6. 验证与排障

### 6.1 分段验证

| 现象 | 结论 |
|---|---|
| Unity Console 没有 `hub=up` 那行 | 组件没在跑。检查 GameObject 是否激活、组件是否勾选、场景是否保存 |
| 有 `hub=up`,但 Console 出现 `not sending: no ergonomics data` | 手套休眠或 Manus Core 没在推数据。**Manus 手套闲置会自动休眠**,动一动唤醒 |
| 有 `hub=up`,Console 出现 `all 20 RIGHT channels are zero` | 戴的是另一只手套。Inspector 里改 `Right Hand` 勾选(Play 模式下可以直接改,立即生效) |
| Unity 正常发包,Ubuntu 收不到 | 网络问题。见下 |
| 一切正常但切到别的窗口就断 | 没加 `Application.runInBackground = true` |

### 6.2 网络排障

先用 PowerShell 直接发包,**绕开 Unity**,确认是网络还是 Unity 的问题:

```powershell
$c = New-Object System.Net.Sockets.UdpClient
$c.EnableBroadcast = $true
$b = [Text.Encoding]::ASCII.GetBytes('{"ergo":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}')
1..60 | ForEach-Object { $c.Send($b, $b.Length, "192.168.1.255", 9881) | Out-Null; Start-Sleep -Milliseconds 100 }
$c.Close()
```

Ubuntu 那边收到 → 网络通,问题在 Unity。收不到 → 检查防火墙、两台机是否真在同一网段。

**双网卡同网段的坑:** 如果 Ubuntu 有两块网卡都在 `192.168.1.0/24`(比如有线 `.10` + 无线 `.36`),而 Windows 只在其中一侧,Ubuntu 主动发起的方向要手动加路由:

```bash
sudo ip route replace 192.168.1.17/32 dev wlo1 src 192.168.1.36
```

(接收方向不受影响 —— 手套数据是单向 Windows→Ubuntu。)

### 6.3 Unity 侧的两个日常陷阱

**改任何 `Assets/` 下的文件都会触发重编译,并自动退出 Play 模式。** 改完脚本记得重新按 ▶。

**Play 模式下创建的 GameObject,退出 Play 时会被销毁。** 一定要在非 Play 状态下建好、Ctrl+S 存场景。

---

## 7. 实测性能

| 指标 | 实测 |
|---|---|
| Unity → Ubuntu 包率 | 64–90 包/秒(由 `sendRateHz` 设定,实测能跑满) |
| 单包大小 | 约 160 字节 ASCII |
| 延迟 | 一帧量级,未单独测量 |
| 稳定性 | 连续运行无丢流(前提是 `runInBackground` 已开、手套未休眠) |

---

## 8. 扩展:3D 骨架

除了 20 个关节角,Manus 插件还能给**完整的 3D 手部骨架**(25 个节点,每个带位置和四元数),订阅方式:

```csharp
ManusManager.communicationHub.onRawSkeletonData.AddListener(OnRawSkeleton);
// 节点语义(哪根手指、哪一节)另外查:
ManusManager.communicationHub.GetRawSkeletonNodeInfo(gloveId, out CoreSDK.NodeInfo[] info);
```

⚠️ **原始骨架是父子相对的位姿树,不是世界坐标。** 每个非掌骨节点的位置读出来都是 `(0, 0, 骨节长度)`,x=y=0 —— 必须连同旋转和 `parentId` 一起取,在接收端做正运动学才能得到真实 3D 位置。

判据:合成正确的话,腕→中指尖约 0.18–0.21 m,食指根→小指根约 0.06–0.08 m。

本仓库的 `ManusSkeletonBridge.cs` 和 `manus_skel_node.py` 实现了这条通路,可作参考。

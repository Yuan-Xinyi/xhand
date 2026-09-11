# Manus -> XHand teleop bridge (Windows side)

The Windows machine only forwards raw numbers; all mapping/calibration happens
on Linux, so this side stays tiny and never needs to change.

## What to send

UDP to `<linux-host>:9881`, one JSON object per glove update (~60-120 Hz is
fine, the Linux loop runs at 20 Hz and takes the freshest):

```json
{"ergo": [20 floats, degrees, Manus ergonomics order]}
```

Order (Manus `ErgonomicsDataType`, one hand):

```
thumb: CMCSpread, CMCStretch, MCPStretch, IPStretch
index: MCPSpread, MCPStretch, PIPStretch, DIPStretch
middle: MCPSpread, MCPStretch, PIPStretch, DIPStretch
ring:  MCPSpread, MCPStretch, PIPStretch, DIPStretch
pinky: MCPSpread, MCPStretch, PIPStretch, DIPStretch
```

Send the RIGHT hand (the XHand is a right hand).

## Option A — patch the Manus SDK sample (recommended)

In `SDKMinimalClient` (or `SDKClient`), the ergonomics callback already hands
you the 20 values. Add a UDP socket and one send per callback:

```cpp
// --- once, near the top -----------------------------------------------
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
static SOCKET g_sock = INVALID_SOCKET;
static sockaddr_in g_dst{};
static void BridgeInit(const char* ip, unsigned short port) {
    WSADATA w; WSAStartup(MAKEWORD(2,2), &w);
    g_sock = socket(AF_INET, SOCK_DGRAM, 0);
    g_dst.sin_family = AF_INET; g_dst.sin_port = htons(port);
    g_dst.sin_addr.s_addr = inet_addr(ip);
}

// --- inside OnErgonomicsCallback, for the RIGHT hand -------------------
char buf[1024];
int n = snprintf(buf, sizeof(buf), "{\"ergo\":[");
for (int i = 0; i < 20; ++i)
    n += snprintf(buf + n, sizeof(buf) - n, "%s%.3f", i ? "," : "",
                  p_Ergo->data[i]);          // 20 values of this hand
n += snprintf(buf + n, sizeof(buf) - n, "]}");
sendto(g_sock, buf, n, 0, (sockaddr*)&g_dst, sizeof(g_dst));
```

Call `BridgeInit("<linux-ip>", 9881);` once at startup.

## Option B — any other source

If you already stream glove data into Python/C#/Unity on Windows, just send the
same JSON. Joint names also work instead of the array:

```json
{"joints": {"index_mcp": 0.9, "index_pip": 1.2, "thumb_cmc": 0.5, ...}}
```

(radians, 0 = straight). Missing joints hold their last value.

## Check the link

On Linux, before touching the robot:

```bash
python RealExperiments/manus_bridge/glove_monitor.py
```

It prints the live joint angles it receives. Once numbers move with your hand,
run the calibration + teleop.

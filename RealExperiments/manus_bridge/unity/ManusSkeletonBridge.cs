// Streams the MANUS raw hand skeleton (3D nodes) to the XHand teleop harness.
//
// Why this instead of ManusUdpBridge's 20 ergonomics angles: those had to be
// hand-assigned to the 12 robot joints with a sign and a range each, and every
// wrong guess could only be caught by watching the real hand.  3D joint
// positions feed the DexPilot retargeter the camera path already uses, where
// the mapping falls out of geometry -- and it can be checked in sim first.
//
// Setup (Unity):
//   1. Empty GameObject -> Add Component -> Manus Skeleton Bridge
//   2. Leave Linux Host on the .255 broadcast address, press Play
//   3. Disable any ManusUdpBridge component on the same scene
//
// Wire format (matches tools/teleop/manus_skel_node.py), one datagram per frame:
//   {"skel":[[x,y,z],...],"meta":[[chainType,fingerJointType],...]}
//
// meta rides along every frame on purpose: it is small, and it lets the Linux
// side work out which node is which without this file having to encode any
// assumption about MANUS's node ordering.
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Manus;

public class ManusSkeletonBridge : MonoBehaviour
{
    [Tooltip("Where manus_skel_node.py listens. Broadcast (x.y.z.255) survives a " +
             "DHCP lease change on the Linux box.")]
    public string linuxHost = "192.168.1.255";
    public int linuxPort = 9882;
    [Tooltip("XHand is a right hand")]
    public bool rightHand = true;
    public float sendRateHz = 60f;
    public bool debugLog = true;

    UdpClient _udp;
    IPEndPoint _dst;
    float _next, _nextReport;
    int _sent;
    CoreSDK.RawSkeletonStream _stream;
    bool _haveStream;
    readonly StringBuilder _sb = new StringBuilder(4096);

    void Start()
    {
        // Otherwise the editor throttles Update() whenever its window is not
        // focused -- i.e. whenever the operator is looking at the robot.
        Application.runInBackground = true;

        var hub = ManusManager.communicationHub;
        _udp = new UdpClient();
        _udp.EnableBroadcast = true;
        _dst = new IPEndPoint(IPAddress.Parse(linuxHost), linuxPort);
        Debug.Log($"[ManusSkeletonBridge] hub={(hub != null ? "up" : "NULL")}, " +
                  $"sending {(rightHand ? "RIGHT" : "LEFT")} skeleton to {linuxHost}:{linuxPort}");
    }

    void OnEnable()
    {
        ManusManager.communicationHub.onRawSkeletonData.AddListener(OnRawSkeleton);
    }

    void OnDisable()
    {
        ManusManager.communicationHub.onRawSkeletonData.RemoveListener(OnRawSkeleton);
    }

    void OnRawSkeleton(CoreSDK.RawSkeletonStream p_Stream)
    {
        _stream = p_Stream;
        _haveStream = true;
    }

    void Update()
    {
        if (Time.unscaledTime < _next) return;
        _next = Time.unscaledTime + 1f / Mathf.Max(1f, sendRateHz);

        if (!_haveStream || _stream.skeletons == null || _stream.skeletons.Count == 0)
        {
            Report("no raw skeleton stream (gloves asleep, or Manus Core not streaming)");
            return;
        }

        foreach (var skel in _stream.skeletons)
        {
            if (skel.nodes == null || skel.nodes.Length == 0) continue;

            CoreSDK.NodeInfo[] info;
            if (!ManusManager.communicationHub.GetRawSkeletonNodeInfo(skel.gloveId, out info)
                || info == null || info.Length != skel.nodes.Length)
            {
                Report($"glove {skel.gloveId}: {skel.nodes.Length} nodes but node info " +
                       $"{(info == null ? "missing" : info.Length.ToString())}");
                continue;
            }

            // One glove per side; skip the one we are not driving.
            var wantSide = rightHand ? CoreSDK.Side.Right : CoreSDK.Side.Left;
            bool sideMatches = false;
            for (int i = 0; i < info.Length; i++)
                if (info[i].side == wantSide) { sideMatches = true; break; }
            if (!sideMatches) continue;

            _sb.Clear();
            _sb.Append("{\"skel\":[");
            for (int i = 0; i < skel.nodes.Length; i++)
            {
                var p = skel.nodes[i].transform.position;
                if (i > 0) _sb.Append(',');
                _sb.Append('[').Append(F(p.x)).Append(',').Append(F(p.y))
                   .Append(',').Append(F(p.z)).Append(']');
            }
            _sb.Append("],\"meta\":[");
            for (int i = 0; i < info.Length; i++)
            {
                if (i > 0) _sb.Append(',');
                _sb.Append('[').Append((int)info[i].chainType).Append(',')
                   .Append((int)info[i].fingerJointType).Append(']');
            }
            _sb.Append("]}");

            byte[] bytes = Encoding.ASCII.GetBytes(_sb.ToString());
            _udp.Send(bytes, bytes.Length, _dst);
            _sent++;
            if (debugLog && _sent % 120 == 0)
                Debug.Log($"[ManusSkeletonBridge] {_sent} packets, {skel.nodes.Length} nodes, " +
                          $"{bytes.Length} bytes");
            return;   // one glove per frame is enough
        }
    }

    static string F(float v)
    {
        return v.ToString("F5", CultureInfo.InvariantCulture);
    }

    // Throttled so a persistent fault states itself once a second rather than
    // at frame rate.
    void Report(string why)
    {
        if (!debugLog || Time.unscaledTime < _nextReport) return;
        _nextReport = Time.unscaledTime + 1f;
        Debug.LogWarning($"[ManusSkeletonBridge] not sending: {why}");
    }

    void OnDestroy()
    {
        _udp?.Close();
        Debug.Log($"[ManusSkeletonBridge] stopped after {_sent} packets");
    }
}

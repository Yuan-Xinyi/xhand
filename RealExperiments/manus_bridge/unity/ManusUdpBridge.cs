// Forwards MANUS glove ergonomics to the XHand teleop harness on Linux.
//
// Setup (Unity):
//   1. Import ManusUnityPlugin_v3.1.1.unitypackage
//   2. Put this file in Assets/  (any folder)
//   3. Create an empty GameObject, add this component
//   4. Set "Linux Host" to the Linux box IP, press Play
//
// Wire format (matches RealExperiments/teleop_repose.py --source udp):
//   {"ergo":[20 floats, degrees, MANUS order for ONE hand]}
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Manus;          // CommunicationHub
using Manus.Utility;  // (harmless if unused)

public class ManusUdpBridge : MonoBehaviour
{
    [Tooltip("IP of the Linux machine running teleop_repose.py")]
    public string linuxHost = "192.168.1.33";
    public int linuxPort = 9881;
    [Tooltip("XHand is a right hand; uncheck only to stream the left glove")]
    public bool rightHand = true;
    [Tooltip("Send at most this many packets per second")]
    public float sendRateHz = 90f;
    [Tooltip("Log the first values to the Console so you can verify movement")]
    public bool debugLog = false;

    UdpClient _udp;
    IPEndPoint _dst;
    float _next;
    int _sent;
    readonly StringBuilder _sb = new StringBuilder(512);

    void Start()
    {
        _udp = new UdpClient();
        _dst = new IPEndPoint(IPAddress.Parse(linuxHost), linuxPort);
        Debug.Log($"[ManusUdpBridge] sending {(rightHand ? "RIGHT" : "LEFT")} hand ergonomics to {linuxHost}:{linuxPort}");
    }

    void Update()
    {
        if (Time.unscaledTime < _next) return;
        _next = Time.unscaledTime + 1f / Mathf.Max(1f, sendRateHz);

        var stream = CommunicationHub.ergonomicsData;
        if (stream.data == null || stream.data.Count == 0) return;

        int offset = rightHand ? 20 : 0;
        foreach (var ergo in stream.data)
        {
            if (ergo.isUserID || ergo.data == null || ergo.data.Length < offset + 20) continue;

            bool any = false;
            for (int i = 0; i < 20; i++)
                if (Mathf.Abs(ergo.data[offset + i]) > 1e-4f) { any = true; break; }
            if (!any) continue;   // this glove has no data for the requested hand

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
                Debug.Log($"[ManusUdpBridge] {_sent} packets, thumbStretch={ergo.data[offset + 1]:F1} indexStretch={ergo.data[offset + 5]:F1}");
            return;   // one glove per frame is enough
        }
    }

    void OnDestroy()
    {
        _udp?.Close();
        Debug.Log($"[ManusUdpBridge] stopped after {_sent} packets");
    }
}

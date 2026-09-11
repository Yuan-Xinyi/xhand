// Forwards MANUS glove ergonomics to the XHand teleop harness on Linux.
//
// Setup (Unity):
//   1. Import ManusUnityPlugin_v3.1.1.unitypackage
//   2. Put this file in Assets/  (any folder, filename must be ManusUdpBridge.cs)
//   3. Create an empty GameObject, add this component
//   4. Set "Linux Host" to the Linux box IP, press Play
//
// Nothing else goes in the scene: CommunicationHub is not a MonoBehaviour ("this
// component should not be added to the scene manually") and ManusManager is
// [InitializeOnLoad], so the editor spins it up on its own.
//
// Wire format (matches tools/teleop/manus_node.py):
//   {"ergo":[20 floats, degrees, MANUS channel order for ONE hand]}
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Manus;          // CommunicationHub / ManusManager

public class ManusUdpBridge : MonoBehaviour
{
    [Tooltip("IP of the Linux machine running manus_node.py")]
    public string linuxHost = "192.168.1.33";
    public int linuxPort = 9881;
    [Tooltip("XHand is a right hand; uncheck only to stream the left glove")]
    public bool rightHand = true;
    [Tooltip("Send at most this many packets per second")]
    public float sendRateHz = 90f;
    [Tooltip("Log progress to the Console so you can verify movement")]
    public bool debugLog = true;

    UdpClient _udp;
    IPEndPoint _dst;
    float _next;
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
        if (stream.data == null || stream.data.Count == 0) return;

        int offset = rightHand ? 20 : 0;
        foreach (var ergo in stream.data)
        {
            if (ergo.isUserID || ergo.data == null || ergo.data.Length < offset + 20) continue;

            bool any = false;
            for (int i = 0; i < 20; i++)
                if (Mathf.Abs(ergo.data[offset + i]) > 1e-4f) { any = true; break; }
            if (!any) continue;   // this glove carries no data for the requested hand

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

    void OnDestroy()
    {
        _udp?.Close();
        Debug.Log($"[ManusUdpBridge] stopped after {_sent} packets");
    }
}

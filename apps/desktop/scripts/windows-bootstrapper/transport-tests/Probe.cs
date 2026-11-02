using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

sealed class FixtureHandler : HttpMessageHandler {
    public bool FailHttp3;
    public bool StallHttp3;
    public int Calls;
    public Uri LastUri;
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancel) {
        Calls++;
        LastUri = request.RequestUri;
        if (request.Version.Major == 3 && FailHttp3) throw new HttpRequestException("QUIC unavailable");
        if (request.Version.Major == 3 && StallHttp3) return WaitForCancellation(cancel);
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Version = request.Version, RequestMessage = request });
    }
    static async Task<HttpResponseMessage> WaitForCancellation(CancellationToken cancel) {
        await Task.Delay(Timeout.Infinite, cancel);
        throw new Exception("Expected cancellation");
    }
}
static class Probe {
    static int Main(string[] args) {
        if (args.Length == 1 && args[0] == "--live") {
            var endpoint = new Uri("https://download.semantier.cn/healthz");
            using (var handler = new SocketsHttpHandler { AllowAutoRedirect = false, UseProxy = false })
            using (var client = new HttpClient(handler)) {
                client.Timeout = TimeSpan.FromSeconds(20);
                using (var h3Response = BootstrapperTransport.Send(client, endpoint, () => 443)) {
                    if (h3Response.StatusCode != HttpStatusCode.OK || h3Response.Version.Major != 3) throw new Exception("Live HTTP/3 unavailable");
                    Console.WriteLine("LIVE_HTTP3_OK " + h3Response.Version);
                }
                using (var tcpResponse = BootstrapperTransport.Send(client, endpoint, () => throw new InvalidOperationException("simulate unavailable HTTPS record"))) {
                    if (tcpResponse.StatusCode != HttpStatusCode.OK || tcpResponse.Version.Major > 2) throw new Exception("Live TCP fallback unavailable");
                    Console.WriteLine("LIVE_TCP_FALLBACK_OK " + tcpResponse.Version);
                }
            }
            return 0;
        }
        Uri origin = new Uri("https://download.semantier.cn/package.zip");
        var h3 = new FixtureHandler();
        using (var client = new HttpClient(h3)) using (var r = BootstrapperTransport.Send(client, origin, () => 443)) {
            if (r.Version.Major != 3 || h3.Calls != 1) throw new Exception("HTTP/3 preference failed");
        }
        var down = new FixtureHandler { FailHttp3 = true };
        using (var client = new HttpClient(down)) using (var r = BootstrapperTransport.Send(client, origin, () => 8444)) {
            if (r.Version.Major != 2 || down.Calls != 2 || down.LastUri != origin) throw new Exception("TCP fallback failed");
        }
        var missing = new FixtureHandler();
        using (var client = new HttpClient(missing)) using (var r = BootstrapperTransport.Send(client, origin, () => throw new InvalidOperationException("DNS unavailable"))) {
            if (r.Version.Major != 2 || missing.Calls != 1 || missing.LastUri != origin) throw new Exception("DNS fallback failed");
        }
        var stalled = new FixtureHandler { StallHttp3 = true };
        using (var client = new HttpClient(stalled)) using (var r = BootstrapperTransport.Send(client, origin, () => 443, TimeSpan.FromMilliseconds(50))) {
            if (r.Version.Major != 2 || stalled.Calls != 2 || stalled.LastUri != origin) throw new Exception("QUIC timeout fallback failed");
        }
        Console.WriteLine("TRANSPORT_POLICY_TEST_PASS");
        return 0;
    }
}

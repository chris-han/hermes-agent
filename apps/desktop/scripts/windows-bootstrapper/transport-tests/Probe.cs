using System;
using System.Net;
using System.Net.Http;
using System.IO;
using System.Security.Cryptography;
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
sealed class ResumeHandler : HttpMessageHandler {
    public bool CorruptRange;
    public int Calls;
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancel) {
        Calls++;
        var second = Calls > 1;
        if (second && request.Headers.Range?.ToString() != "bytes=3-") throw new Exception("Missing resume range");
        var response = new HttpResponseMessage(second ? HttpStatusCode.PartialContent : HttpStatusCode.OK) {
            Version = request.Version, RequestMessage = request,
            Content = new ByteArrayContent(second ? new byte[] {4,5,6} : new byte[] {1,2,3})
        };
        response.Content.Headers.ContentLength = second ? 3 : 6;
        if (second) response.Content.Headers.ContentRange = new System.Net.Http.Headers.ContentRangeHeaderValue(CorruptRange ? 2 : 3, 5, 6);
        return Task.FromResult(response);
    }
}
sealed class EmptyHandler : HttpMessageHandler {
    public int Calls;
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancel) {
        Calls++;
        var response = new HttpResponseMessage(HttpStatusCode.OK) { Version = request.Version, Content = new ByteArrayContent(Array.Empty<byte>()) };
        response.Content.Headers.ContentLength = 6;
        return Task.FromResult(response);
    }
}
static class Probe {
    static int Main(string[] args) {
        if (args.Length == 1 && args[0] == "--live") {
            var endpoint = new Uri("https://download.semantier.cn/healthz");
            using (var handler = new SocketsHttpHandler { AllowAutoRedirect = false, UseProxy = false })
            using (var client = new HttpClient(handler)) {
                client.Timeout = TimeSpan.FromSeconds(20);
                using (var h3Response = BootstrapperTransport.Send(client, endpoint)) {
                    if (h3Response.StatusCode != HttpStatusCode.OK || h3Response.Version.Major != 3) throw new Exception("Live HTTP/3 unavailable");
                    Console.WriteLine("LIVE_HTTP3_OK " + h3Response.Version);
                }
                var file = new Uri("https://download.semantier.cn/releases/channel-builds/20bceda1fd864ca7a9b9dd67fa4879c3/semantier-setup-channel1.exe");
                using (var full = new MemoryStream()) {
                    BootstrapperTransport.Download(client, file, full, 476664, _ => {});
                    if (Convert.ToHexString(SHA256.HashData(full.ToArray())).ToLowerInvariant() != "7168c79a77c760161bb2bfbca953207d6bb83808102412e1886d5811acdfea38") throw new Exception("Live fixture hash mismatch");
                    using (var part = BootstrapperTransport.Send(client, file, offset: 1024)) {
                        var range = part.Content.Headers.ContentRange;
                        var received = part.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();
                        var all = full.ToArray();
                        if (part.Version.Major != 3 || part.StatusCode != HttpStatusCode.PartialContent || range?.From != 1024 || range.Length != all.Length || received.Length != all.Length - 1024 ||
                            !SHA256.HashData(received).AsSpan().SequenceEqual(SHA256.HashData(all.AsSpan(1024)))) throw new Exception("Live HTTP/3 range mismatch");
                        Console.WriteLine("LIVE_HTTP3_RANGE_OK " + received.Length);
                    }
                }
            }
            return 0;
        }
        Uri origin = new Uri("https://download.semantier.cn/package.zip");
        var h3 = new FixtureHandler();
        using (var client = new HttpClient(h3)) using (var r = BootstrapperTransport.Send(client, origin)) {
            if (r.Version.Major != 3 || h3.Calls != 1) throw new Exception("HTTP/3 preference failed");
        }
        foreach (var invalid in new[] { "https://download.semantier.cn:8444/package.zip", "http://download.semantier.cn/package.zip" }) {
            bool rejected = false;
            using (var client = new HttpClient(new FixtureHandler())) {
                try { BootstrapperTransport.Send(client, new Uri(invalid)); } catch (InvalidOperationException) { rejected = true; }
            }
            if (!rejected) throw new Exception("Unsafe transport origin accepted");
        }
        var down = new FixtureHandler { FailHttp3 = true };
        using (var client = new HttpClient(down)) using (var r = BootstrapperTransport.Send(client, origin)) {
            if (r.Version.Major != 2 || down.Calls != 2 || down.LastUri != origin) throw new Exception("TCP fallback failed");
        }
        var stalled = new FixtureHandler { StallHttp3 = true };
        using (var client = new HttpClient(stalled)) using (var r = BootstrapperTransport.Send(client, origin, TimeSpan.FromMilliseconds(50))) {
            if (r.Version.Major != 2 || stalled.Calls != 2 || stalled.LastUri != origin) throw new Exception("QUIC timeout fallback failed");
        }
        var resumed = new ResumeHandler();
        using (var client = new HttpClient(resumed)) using (var output = new MemoryStream()) {
            BootstrapperTransport.Download(client, origin, output, 6, _ => {});
            if (resumed.Calls != 2 || Convert.ToHexString(output.ToArray()) != "010203040506") throw new Exception("Resume failed");
        }
        var corrupt = new ResumeHandler { CorruptRange = true };
        using (var client = new HttpClient(corrupt)) using (var output = new MemoryStream()) {
            bool rejected = false;
            try { BootstrapperTransport.Download(client, origin, output, 6, _ => {}); } catch (InvalidOperationException) { rejected = true; }
            if (!rejected || output.Length != 3) throw new Exception("Incorrect range accepted");
        }
        var empty = new EmptyHandler();
        using (var client = new HttpClient(empty)) using (var output = new MemoryStream()) {
            bool rejected = false;
            try { BootstrapperTransport.Download(client, origin, output, 6, _ => {}); } catch (IOException) { rejected = true; }
            if (!rejected || empty.Calls != 5 || output.Length != 0) throw new Exception("Incomplete download was not bounded");
        }
        Console.WriteLine("TRANSPORT_POLICY_TEST_PASS");
        return 0;
    }
}

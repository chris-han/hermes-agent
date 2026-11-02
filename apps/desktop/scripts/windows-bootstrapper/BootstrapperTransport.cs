using System;
using System.Net.Http;
using System.Threading;

// Prefer QUIC. A network failure or missing HTTPS service record falls back to
// the same HTTPS origin on its normal TCP port; neither path follows redirects.
static class BootstrapperTransport {
    internal static HttpResponseMessage Send(HttpClient client, Uri origin, Func<int> http3Port, TimeSpan? quicTimeout = null) {
        int port;
        try { port = http3Port(); }
        catch (InvalidOperationException) { return Tcp(client, origin); }
        var h3Uri = new UriBuilder(origin) { Port = port }.Uri;
        using (var deadline = new CancellationTokenSource(quicTimeout ?? TimeSpan.FromSeconds(15))) try {
            using (var request = new HttpRequestMessage(HttpMethod.Get, h3Uri)) {
                request.Version = new Version(3, 0);
                request.VersionPolicy = HttpVersionPolicy.RequestVersionExact;
                return client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token).GetAwaiter().GetResult();
            }
        } catch (HttpRequestException) {
            return Tcp(client, origin);
        } catch (OperationCanceledException) when (deadline.IsCancellationRequested) {
            return Tcp(client, origin);
        }
    }

    static HttpResponseMessage Tcp(HttpClient client, Uri origin) {
        using (var request = new HttpRequestMessage(HttpMethod.Get, origin)) {
            request.Version = new Version(2, 0);
            request.VersionPolicy = HttpVersionPolicy.RequestVersionOrLower;
            return client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        }
    }
}

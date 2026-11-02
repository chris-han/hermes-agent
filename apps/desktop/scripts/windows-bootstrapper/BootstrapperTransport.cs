using System;
using System.Net.Http;
using System.Threading;

// Prefer QUIC on the normal HTTPS port. A network failure falls back to the
// same HTTPS origin over TCP; neither path follows redirects.
static class BootstrapperTransport {
    internal static HttpResponseMessage Send(HttpClient client, Uri origin, TimeSpan? quicTimeout = null) {
        if (origin.Scheme != Uri.UriSchemeHttps || origin.Port != 443) throw new InvalidOperationException("HTTPS port 443 required.");
        using (var deadline = new CancellationTokenSource(quicTimeout ?? TimeSpan.FromSeconds(15))) try {
            using (var request = new HttpRequestMessage(HttpMethod.Get, origin)) {
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

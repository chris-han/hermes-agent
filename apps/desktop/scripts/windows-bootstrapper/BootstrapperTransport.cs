using System;
using System.Net.Http;
using System.Net;
using System.Net.Http.Headers;
using System.IO;
using System.Threading;

// Prefer QUIC on the normal HTTPS port. A network failure falls back to the
// same HTTPS origin over TCP; neither path follows redirects.
static class BootstrapperTransport {
    internal static HttpResponseMessage Send(HttpClient client, Uri origin, TimeSpan? quicTimeout = null, long offset = 0, bool preferHttp3 = true) {
        if (origin.Scheme != Uri.UriSchemeHttps || origin.Port != 443) throw new InvalidOperationException("HTTPS port 443 required.");
        if (offset < 0) throw new InvalidOperationException("Invalid range offset.");
        if (!preferHttp3) return Tcp(client, origin, offset);
        using (var deadline = new CancellationTokenSource(quicTimeout ?? TimeSpan.FromSeconds(15))) try {
            using (var request = new HttpRequestMessage(HttpMethod.Get, origin)) {
                if (offset > 0) request.Headers.Range = new RangeHeaderValue(offset, null);
                request.Version = new Version(3, 0);
                request.VersionPolicy = HttpVersionPolicy.RequestVersionExact;
                return client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token).GetAwaiter().GetResult();
            }
        } catch (HttpRequestException) {
            return Tcp(client, origin, offset);
        } catch (OperationCanceledException) when (deadline.IsCancellationRequested) {
            return Tcp(client, origin, offset);
        }
    }

    static HttpResponseMessage Tcp(HttpClient client, Uri origin, long offset) {
        using (var request = new HttpRequestMessage(HttpMethod.Get, origin)) {
            if (offset > 0) request.Headers.Range = new RangeHeaderValue(offset, null);
            request.Version = new Version(2, 0);
            request.VersionPolicy = HttpVersionPolicy.RequestVersionOrLower;
            return client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        }
    }

    internal static void Download(HttpClient client, Uri origin, Stream output, long size, Action<long> progress, bool preferHttp3 = true) {
        if (size <= 0 || !output.CanWrite || output.Position != 0) throw new InvalidOperationException("Invalid download target.");
        long total = 0;
        var bytes = new byte[1024 * 1024];
        for (int attempt = 0; attempt < 5 && total < size; attempt++) {
            HttpResponseMessage response;
            try { response = Send(client, origin, offset: total, preferHttp3: preferHttp3); }
            catch (HttpRequestException) { if (attempt == 4) throw; continue; }
            using (response) {
                var range = response.Content.Headers.ContentRange;
                if (response.Version.Major < 1 || response.Version.Major > 3 ||
                    response.Content.Headers.ContentLength != size - total ||
                    (total == 0 ? response.StatusCode != HttpStatusCode.OK :
                        response.StatusCode != HttpStatusCode.PartialContent || range == null ||
                        range.Unit != "bytes" || range.From != total || range.To != size - 1 || range.Length != size))
                    throw new InvalidOperationException("Unexpected download response.");
                using (var source = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()) {
                    while (total < size) {
                        int n;
                        try { n = source.Read(bytes, 0, (int)Math.Min(bytes.Length, size - total)); }
                        catch (IOException) { break; }
                        if (n == 0) break;
                        output.Write(bytes, 0, n);
                        total += n;
                        progress(total);
                    }
                }
            }
        }
        if (total != size) throw new IOException("Download incomplete after retries.");
        output.Flush();
    }
}

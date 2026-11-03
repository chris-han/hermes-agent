using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Xml;
using System.Xml.Linq;

// Called only after SHA-256 and Windows Authenticode verification of every file.
static class PackageIdentity {
    const string Identity="Semantier.Agent.Acceptance";
    const string Publisher="CN=Semantier Technology, O=Semantier Technology";
    internal static void Verify(ActiveRelease release, string dir) {
        var rootName=Path.GetFileName(release.Root);
        var expected=new HashSet<string>(release.Objects.Where(x=>x.Key!=release.Root).Select(x=>Path.GetFileName(x.Key)),StringComparer.OrdinalIgnoreCase);
        if(expected.Count!=release.Objects.Count-1 || expected.Any(x=>!x.EndsWith(".msix",StringComparison.OrdinalIgnoreCase))) throw new InvalidOperationException("Ambiguous package closure.");
        foreach(var item in release.Objects) {
            string file=Path.GetFileName(item.Key); bool bundle=file==rootName;
            using(var archive=ZipFile.OpenRead(Path.Combine(dir,file))) {
                string path=bundle ? "AppxMetadata/AppxBundleManifest.xml" : "AppxManifest.xml";
                var entries=archive.Entries.Where(x=>x.FullName==path).ToArray();
                if(entries.Length!=1 || entries[0].Length>4*1024*1024) throw new InvalidOperationException("Package identity manifest mismatch.");
                using(var input=entries[0].Open()) using(var reader=XmlReader.Create(input,new XmlReaderSettings{DtdProcessing=DtdProcessing.Prohibit,XmlResolver=null,MaxCharactersInDocument=4*1024*1024})) {
                    var document=XDocument.Load(reader); var root=document.Root;
                    string ns=root?.Name.NamespaceName;
                    if(root==null || (bundle ? root.Name.LocalName!="Bundle" || (ns!="http://schemas.microsoft.com/appx/2017/bundle" && ns!="http://schemas.microsoft.com/appx/2013/bundle") : root.Name.LocalName!="Package" || ns!="http://schemas.microsoft.com/appx/manifest/foundation/windows10")) throw new InvalidOperationException("Package manifest namespace mismatch.");
                    var identity=root.Elements().SingleOrDefault(x=>x.Name==root.Name.Namespace+"Identity");
                    string name=(string)identity?.Attribute("Name"), publisher=(string)identity?.Attribute("Publisher"), version=(string)identity?.Attribute("Version");
                    if((name!=Identity && !(name?.StartsWith(Identity+".",StringComparison.Ordinal)??false)) || publisher!=Publisher || version!=release.Version.ToString()) throw new InvalidOperationException("Package identity mismatch.");
                    if(bundle) {
                        var references=root.Descendants().Where(x=>x.Name==root.Name.Namespace+"Package").Select(x=>(string)x.Attribute("FileName")).ToList();
                        if(references.Any(x=>string.IsNullOrEmpty(x) || x!=Path.GetFileName(x)) ||
                           references.Count!=expected.Count || new HashSet<string>(references,StringComparer.OrdinalIgnoreCase).Count!=references.Count ||
                           !expected.SetEquals(references)) throw new InvalidOperationException("Bundle package closure mismatch.");
                    }
                }
            }
        }
    }
}

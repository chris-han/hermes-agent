using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;

static class PackageProbe {
    static ActiveRelease Make(JsonElement graph,bool missing,string version) {
        var objects=graph.GetProperty("objects").EnumerateArray().Select(x=>new ActiveRelease.Object(x.GetProperty("key").GetString(),x.GetProperty("size").GetInt64(),x.GetProperty("sha256").GetString())).ToArray();
        if(missing) objects=objects.Take(objects.Length-1).ToArray();
        string root=graph.GetProperty("root").GetString();
        var release=(ActiveRelease)Activator.CreateInstance(typeof(ActiveRelease),true);
        void Set(string name,object value) => typeof(ActiveRelease).GetProperty(name,BindingFlags.Instance|BindingFlags.NonPublic).SetValue(release,value);
        Set("BuildId",root.Split('/')[2]);Set("Version",new Version(version));Set("Root",root);Set("Objects",objects);
        return release;
    }
    static void Reject(Action action) {try {action();throw new Exception("invalid package set accepted");} catch(InvalidOperationException) {}}
    public static void Main(string[] args) {
        using var doc=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(args[0],"native-graph.json")));
        string version=doc.RootElement.GetProperty("objects")[0].GetProperty("version").GetString();
        PackageIdentity.Verify(Make(doc.RootElement,false,version),args[0]);
        Reject(()=>PackageIdentity.Verify(Make(doc.RootElement,true,version),args[0]));
        Reject(()=>PackageIdentity.Verify(Make(doc.RootElement,false,version=="0.0.0.0" ? "0.0.0.1" : "0.0.0.0"),args[0]));
        Console.WriteLine("PACKAGE_IDENTITY_AND_CLOSURE_PASS");
    }
}

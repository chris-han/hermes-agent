using System;
using System.ComponentModel;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Net;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows.Forms;

// Native adapter of desktop DESIGN.md and src/themes/presets.ts (semantier).
// No WebView, online fonts, or additional runtime dependency.
static class InstallerText {
    static string language = Resolve(CultureInfo.CurrentUICulture.Name);
    public static string Resolve(string culture) {
        try { return new CultureInfo(culture).TwoLetterISOLanguageName == "zh" ? "zh-CN" : "en"; }
        catch(CultureNotFoundException) { return "en"; }
    }
    public static string Language { get { return language; } }
    public static void SetLanguage(string value) {
        if(value != "zh-CN" && value != "en") throw new InvalidOperationException("Unsupported installer language.");
        language = value;
    }
    public static string Choose(string english, string chinese) { return language == "zh-CN" ? chinese : english; }
    public static string Error(Exception error) {
        if(error is OperationCanceledException || (error is Win32Exception && ((Win32Exception)error).NativeErrorCode == 1223))
            return Choose("Installation cancelled. No further installation steps were performed.","安装已取消，未继续执行后续安装步骤。");
        if(error is WebException) return Choose("Download failed. Check your connection and try again.","下载失败，请检查网络连接后重试。");
        if(error.Message.StartsWith("A newer Semantier")) return Choose("A newer version is already installed. It will be kept.","已安装较新版本，将保留现有版本。");
        if(error.Message.Contains("already installed for this user")) return Choose("Semantier Agent 0.0.64.0 is already installed for this account.","当前账户已安装 Semantier Agent 0.0.64.0。");
        if(error.Message.Contains("mismatch") || error.Message.Contains("Unsafe") || error.Message.Contains("changed"))
            return Choose("Verification failed. The installation has stopped. Download a fresh installer from semantier.com.","校验失败，安装已停止。请从 semantier.com 重新下载安装程序。");
        if(error.Message.StartsWith("Certificate") || error.Message.StartsWith("Elevation"))
            return Choose("Publisher trust was not completed. Review the Windows permission request before trying again.","发布者信任未完成，请检查 Windows 权限请求后重试。");
        return Choose("Installation could not finish. Review the technical details before trying again.","安装未能完成，请查看技术详情后重试。");
    }
}

sealed class InstallerWindow : Form {
    // Same native palette roles as semantierTheme, not an installer-specific theme.
    static readonly Color Surface = ColorTranslator.FromHtml("#f3f4f5");
    static readonly Color Ink = ColorTranslator.FromHtml("#1a1b1e");
    static readonly Color Muted = ColorTranslator.FromHtml("#43474e");
    static readonly Color ActionYellow = ColorTranslator.FromHtml("#e8c92d");
    readonly Action<Action<string>> execute;
    readonly Label headline, status, stages, note;
    readonly ProgressBar progress;
    readonly Button primary, details;
    readonly TextBox diagnostics;
    readonly PictureBox logo;
    readonly Icon ownedIcon;
    bool busy, complete;
    string cleanupMessage = "";

    static Label TextLabel(string text, float size, FontStyle style, Color color) {
        return new Label { AutoSize=true, MaximumSize=new Size(560,0), Dock=DockStyle.Top,
            Font=new Font("Segoe UI",size,style), ForeColor=color, Text=text,
            Margin=new Padding(0,0,0,12), UseMnemonic=false };
    }
    public InstallerWindow(Action<Action<string>> operation) {
        execute=operation;
        Text=InstallerText.Choose("Semantier setup","Semantier 安装程序");
        Font=new Font("Segoe UI",10F); ForeColor=Ink; BackColor=Surface;
        AutoScaleDimensions=new SizeF(96F,96F); AutoScaleMode=AutoScaleMode.Dpi;
        ClientSize=new Size(640,470); MinimumSize=new Size(656,509);
        StartPosition=FormStartPosition.CenterScreen; MaximizeBox=false;
        using(var embedded=Assembly.GetExecutingAssembly().GetManifestResourceStream("Semantier.icon.ico")) {
            if(embedded==null) throw new InvalidOperationException("Missing Semantier icon.");
            ownedIcon=new Icon(embedded); Icon=ownedIcon;
        }
        var root=new TableLayoutPanel { Dock=DockStyle.Fill, Padding=new Padding(32),
            ColumnCount=1, RowCount=7, BackColor=Surface };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute,88));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute,28));
        root.RowStyles.Add(new RowStyle(SizeType.Percent,100));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        Controls.Add(root);
        var header=new TableLayoutPanel { Dock=DockStyle.Fill,ColumnCount=2,RowCount=1,Margin=Padding.Empty };
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,76));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));
        logo=new PictureBox { Width=56,Height=56,SizeMode=PictureBoxSizeMode.Zoom,Margin=Padding.Empty };
        using(var image=Assembly.GetExecutingAssembly().GetManifestResourceStream("Semantier.logo.png")) {
            if(image==null) throw new InvalidOperationException("Missing Semantier logo.");
            using(var bitmap=Image.FromStream(image)) logo.Image=new Bitmap(bitmap);
        }
        var brand=new FlowLayoutPanel { FlowDirection=FlowDirection.TopDown,Dock=DockStyle.Fill,WrapContents=false,Margin=Padding.Empty };
        var wordmark=TextLabel("semantier",23F,FontStyle.Bold,Ink); wordmark.Margin=Padding.Empty;
        var edition=TextLabel(InstallerText.Choose("Windows · 0.0.64.0 · Acceptance","Windows · 0.0.64.0 · 验收版"),9F,FontStyle.Regular,Muted);
        edition.Margin=new Padding(0,3,0,0); brand.Controls.Add(wordmark);brand.Controls.Add(edition);
        header.Controls.Add(logo,0,0);header.Controls.Add(brand,1,0);root.Controls.Add(header,0,0);
        headline=TextLabel(InstallerText.Choose("Install Semantier Agent","安装 Semantier Agent"),18F,FontStyle.Bold,Ink);
        root.Controls.Add(headline,0,1);
        status=TextLabel(InstallerText.Choose("Download the verified application and install it for your Windows account.","下载并校验应用，然后安装到您当前的 Windows 账户。"),10F,FontStyle.Regular,Muted);
        status.AccessibleRole=AccessibleRole.StaticText;root.Controls.Add(status,0,2);
        progress=new ProgressBar { Dock=DockStyle.Top,Height=8,Minimum=0,Maximum=100,
            Style=ProgressBarStyle.Continuous,Margin=new Padding(0,4,0,12),
            AccessibleName=InstallerText.Choose("Download progress","下载进度") };
        root.Controls.Add(progress,0,3);
        var body=new TableLayoutPanel { Dock=DockStyle.Fill,ColumnCount=1,RowCount=3,Margin=Padding.Empty };
        body.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        body.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        body.RowStyles.Add(new RowStyle(SizeType.Percent,100));
        stages=TextLabel(InstallerText.Choose("Download → Verify → Install","下载 → 校验 → 安装"),9F,FontStyle.Regular,Muted);
        body.Controls.Add(stages,0,0);
        details=new Button { Text=InstallerText.Choose("Technical details","技术详情"),AutoSize=true,Visible=false,
            FlatStyle=FlatStyle.Flat,Margin=new Padding(0,4,0,6),Anchor=AnchorStyles.Left };
        details.FlatAppearance.BorderSize=0;
        details.Click+=(s,e)=> { diagnostics.Visible=!diagnostics.Visible; };
        body.Controls.Add(details,0,1);
        diagnostics=new TextBox { Multiline=true,ReadOnly=true,ScrollBars=ScrollBars.Vertical,
            Dock=DockStyle.Fill,Visible=false,BackColor=SystemColors.Window,
            AccessibleName=InstallerText.Choose("Technical details","技术详情") };
        body.Controls.Add(diagnostics,0,2);root.Controls.Add(body,0,4);
        note=TextLabel(InstallerText.Choose("2.26 GB download. Windows may ask for administrator permission to trust the publisher.","下载大小 2.26 GB。Windows 可能请求管理员权限以信任发布者。"),9F,FontStyle.Regular,Muted);
        note.Margin=new Padding(0,16,0,20);root.Controls.Add(note,0,5);
        var footer=new FlowLayoutPanel { Dock=DockStyle.Fill,AutoSize=true,FlowDirection=FlowDirection.RightToLeft,WrapContents=false,Margin=Padding.Empty };
        primary=new Button { Text=InstallerText.Choose("Download and install","下载并安装"),AutoSize=true,
            MinimumSize=new Size(164,42),FlatStyle=FlatStyle.Flat,BackColor=ActionYellow,ForeColor=Ink,
            Padding=new Padding(18,6,18,6),Margin=Padding.Empty,UseVisualStyleBackColor=false };
        primary.FlatAppearance.BorderSize=0; primary.Click+=Run;
        footer.Controls.Add(primary);root.Controls.Add(footer,0,6);AcceptButton=primary;
        FormClosing+=(sender,eventArgs)=> {
            if(busy) { eventArgs.Cancel=true; note.Text=InstallerText.Choose("Installation is running. Please wait for the result before closing.","安装正在进行，请等待结果后再关闭。"); }
        };
    }
    void Report(string message) {
        int percent;
        if(message.StartsWith("Downloading ") && message.EndsWith("%") &&
            int.TryParse(message.Substring(12,message.Length-13),out percent)) {
            headline.Text=InstallerText.Choose("Downloading Semantier Agent","正在下载 Semantier Agent");
            status.Text=InstallerText.Choose("Downloading ","正在下载 ")+percent.ToString(CultureInfo.CurrentCulture)+"%";
            progress.Style=ProgressBarStyle.Continuous;progress.Value=Math.Max(0,Math.Min(100,percent));
        } else if(message=="Downloading signed release…") {
            headline.Text=InstallerText.Choose("Downloading Semantier Agent","正在下载 Semantier Agent");
            status.Text=InstallerText.Choose("Connecting to the download service…","正在连接下载服务…");
        } else if(message=="Verifying package and certificate…") {
            headline.Text=InstallerText.Choose("Verifying your download","正在校验下载内容");
            status.Text=InstallerText.Choose("Checking package integrity and the publisher certificate…","正在检查安装包完整性与发布者证书…");
            progress.Style=ProgressBarStyle.Marquee;stages.Text=InstallerText.Choose("Download complete → Verify → Install","下载完成 → 校验 → 安装");
        } else if(message=="Waiting for Windows administrator consent…") {
            headline.Text=InstallerText.Choose("Waiting for Windows permission","正在等待 Windows 授权");
            status.Text=InstallerText.Choose("Respond to the Windows administrator prompt to continue.","请在 Windows 管理员权限提示中确认，以继续安装。");
        } else if(message=="Installing for the current Windows user…") {
            headline.Text=InstallerText.Choose("Installing Semantier Agent","正在安装 Semantier Agent");
            status.Text=InstallerText.Choose("Registering the application for your Windows account…","正在为您当前的 Windows 账户注册应用…");
            progress.Style=ProgressBarStyle.Marquee;stages.Text=InstallerText.Choose("Download complete → Verified → Install","下载完成 → 校验完成 → 安装");
        } else if(message.StartsWith("Semantier Agent 0.0.64.0 installed.") || message.StartsWith("Installation stopped.")) {
            cleanupMessage=message;
        }
    }
    async void Run(object sender,EventArgs eventArgs) {
        if(complete) { Close();return; }
        busy=true;primary.Enabled=false;details.Visible=false;diagnostics.Visible=false;
        cleanupMessage="";progress.Value=0;progress.Style=ProgressBarStyle.Continuous;
        stages.Text=InstallerText.Choose("Download → Verify → Install","下载 → 校验 → 安装");
        status.Text=InstallerText.Choose("Checking your installation…","正在检查当前安装状态…");
        // Progress<T> marshals all producer updates onto this window's UI context,
        // preserving final/error ordering without racing queued BeginInvoke calls.
        var updates=new Progress<string>(Report);
        try {
            await Task.Run(()=>execute(message=>((IProgress<string>)updates).Report(message)));
            Finish();
        } catch(Exception error) { Failed(error); }
        finally { busy=false;primary.Enabled=true; }
    }
    void Finish() {
        complete=true;headline.Text=InstallerText.Choose("Installation complete","安装完成");
        status.Text=InstallerText.Choose("Semantier Agent 0.0.64.0 is ready. Open it from the Windows Start menu.","Semantier Agent 0.0.64.0 已就绪，可从 Windows 开始菜单打开。");
        progress.Style=ProgressBarStyle.Continuous;progress.Value=100;
        stages.Text=InstallerText.Choose("Downloaded → Verified → Installed","下载完成 → 校验完成 → 安装完成");
        primary.Text=InstallerText.Choose("Close","关闭");
        note.Text=InstallerText.Choose("Installed for your Windows account.","已安装到您当前的 Windows 账户。");
        if(cleanupMessage.Contains("Temporary files may remain")) {
            note.Text=InstallerText.Choose("Installed. Some temporary files could not be removed; see technical details.","安装已完成，部分临时文件未能删除，详见技术详情。");
            diagnostics.Text=cleanupMessage;details.Visible=true;
        }
    }
    void Failed(Exception error) {
        complete=false;progress.Style=ProgressBarStyle.Continuous;
        headline.Text=InstallerText.Choose("Installation stopped","安装已停止");
        status.Text=InstallerText.Error(error);
        diagnostics.Text=error.ToString()+"\r\n"+cleanupMessage;details.Visible=true;
        primary.Text=InstallerText.Choose("Try again","重试");
        note.Text=InstallerText.Choose("The installer will check your installed version before retrying.","重试前，安装程序会重新检查已安装版本。");
    }
    protected override void Dispose(bool disposing) {
        if(disposing) { if(logo.Image!=null)logo.Image.Dispose();if(ownedIcon!=null)ownedIcon.Dispose(); }
        base.Dispose(disposing);
    }
#if TESTING
    public static void Preview(string language,string state,string output) {
        InstallerText.SetLanguage(language);
        using(var window=new InstallerWindow(report=>{throw new InvalidOperationException("Preview cannot install.");})) {
            window.Show();Application.DoEvents();
            window.primary.Enabled=(state=="ready" || state=="error" || state=="success");
            if(state=="download")window.Report("Downloading 42%");
            else if(state=="verify")window.Report("Verifying package and certificate…");
            else if(state=="trust")window.Report("Waiting for Windows administrator consent…");
            else if(state=="install")window.Report("Installing for the current Windows user…");
            else if(state=="success")window.Finish();
            else if(state=="error")window.Failed(new WebException("Fixture connection failure."));
            else if(state!="ready")throw new InvalidOperationException("Unknown preview state.");
            window.Refresh();Application.DoEvents();
            var paintUntil=DateTime.UtcNow.AddMilliseconds(400);
            while(DateTime.UtcNow<paintUntil) { Application.DoEvents();System.Threading.Thread.Sleep(10); }
            using(var bitmap=new Bitmap(window.Width,window.Height)) {
                window.DrawToBitmap(bitmap,new Rectangle(0,0,bitmap.Width,bitmap.Height));
                bitmap.Save(output,System.Drawing.Imaging.ImageFormat.Png);
            }
            window.Close();
        }
    }
    static void TestExecution(bool fail) {
        bool executed=false;
        Exception assertion=null;
        using(var window=new InstallerWindow(report=>{
            executed=true;report("Downloading 42%");report("Verifying package and certificate…");
            if(fail)throw new WebException("Fixture");
            report("Installing for the current Windows user…");
        }))
        using(var timer=new System.Windows.Forms.Timer { Interval=50 }) {
            DateTime deadline=DateTime.UtcNow.AddSeconds(10);
            timer.Tick+=(sender,eventArgs)=> {
                if(window.busy && DateTime.UtcNow<deadline)return;
                try {
                    if(window.busy || !executed)throw new InvalidOperationException("Fixture producer did not settle.");
                    if(fail) {
                        if(window.complete || !window.details.Visible || window.primary.Text!=InstallerText.Choose("Try again","重试") ||
                            window.status.Text!=InstallerText.Error(new WebException("Fixture")))
                            throw new InvalidOperationException("Queued producer failure/recovery was not preserved.");
                    } else if(!window.complete || window.progress.Value!=100 || window.primary.Text!=InstallerText.Choose("Close","关闭"))
                        throw new InvalidOperationException("Queued producer completion was not preserved.");
                } catch(Exception error) { assertion=error; }
                timer.Stop();window.Close();
            };
            window.Shown+=(sender,eventArgs)=>window.BeginInvoke((Action)(()=>{window.primary.PerformClick();timer.Start();}));
            Application.Run(window);
        }
        if(assertion!=null)throw assertion;
    }
    public static void SelfTest() {
        if(InstallerText.Resolve("zh-CN")!="zh-CN" || InstallerText.Resolve("zh-TW")!="zh-CN" ||
            InstallerText.Resolve("en-US")!="en" || InstallerText.Resolve("de-DE")!="en")
            throw new InvalidOperationException("System UI language resolution failed.");
        foreach(string language in new[]{"zh-CN","en"}) {
            InstallerText.SetLanguage(language);
            using(var window=new InstallerWindow(report=>{throw new InvalidOperationException("Rendering cannot install.");})) {
                window.Report("Downloading 42%");
                if(window.progress.Value!=42 || !window.status.Text.Contains("42%"))throw new InvalidOperationException("Progress binding failed.");
                window.Report("Verifying package and certificate…");
                if(window.progress.Style!=ProgressBarStyle.Marquee)throw new InvalidOperationException("Verification progress failed.");
            }
            TestExecution(true);TestExecution(false);
        }
        InstallerText.SetLanguage(InstallerText.Resolve(CultureInfo.CurrentUICulture.Name));
    }
#endif
}

const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const b=JSON.parse(fs.readFileSync(process.argv[2],'utf8').replace(/^\uFEFF/,''));
const {_electron}=require(b.playwright);fs.mkdirSync(b.output,{recursive:true});let app;const errors=[];
(async()=>{try{
 const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>/^(PATH|SYSTEMROOT|WINDIR|COMSPEC|PATHEXT|USERPROFILE|SYSTEMDRIVE|PROCESSOR_ARCHITECTURE|NUMBER_OF_PROCESSORS|APPDATA|LOCALAPPDATA|TEMP|TMP|PROGRAMFILES|PROGRAMFILES\(X86\))$/i.test(k)));
 app=await _electron.launch({executablePath:b.executable,cwd:path.dirname(b.executable),timeout:120000,
 env:{...env,HERMES_HOME:path.join(b.output,'hermes-home'),HERMES_DESKTOP_USER_DATA_DIR:path.join(b.output,'userdata'),
 HERMES_DESKTOP_IGNORE_EXISTING:'1',HERMES_DESKTOP_SKIP_QUIT_CONFIRM:'1',HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION:'1',HERMES_DESKTOP_CDP_PORT:'off'}});
 app.process().stderr.on('data',x=>{if(String(x).includes('ERR_MODULE_NOT_FOUND'))errors.push(String(x));});
 const facts=await app.evaluate(({app})=>({packaged:app.isPackaged,version:app.getVersion(),executable:process.execPath,platform:process.platform}));
 assert.equal(facts.packaged,true);assert.equal(facts.version,b.version);assert.equal(facts.platform,'win32');assert.equal(path.resolve(facts.executable),path.resolve(b.executable));
 const page=await app.firstWindow({timeout:120000});await page.waitForLoadState('domcontentloaded');assert(await page.isVisible('body'));
 assert(await app.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows().some(w=>w.isVisible())));
 await page.waitForFunction(async()=>{try{const p=await window.hermesDesktop.api({path:'/api/profiles/active'});return typeof p.current==='string'&&p.current.length>0;}catch{return false;}},null,{timeout:180000,polling:2000});
 const pty=await page.evaluate(async()=>{const api=window.hermesDesktop.terminal,s=await api.start({cols:100,rows:30});let off,timer;
 try{return await new Promise((resolve,reject)=>{let output='';timer=setTimeout(()=>reject(Error('PTY deadline exceeded')),30000);
 off=api.onData(s.id,data=>{output+=typeof data==='string'?data:JSON.stringify(data);if(output.includes('SEMANTIER_REAL_PTY_AMD64_OK')){clearTimeout(timer);resolve({shell:s.shell,output});}});
 (async()=>{try{await api.attach(s.id);await api.write(s.id,'cmd.exe /d /c "echo SEMANTIER_REAL_PTY_%PROCESSOR_ARCHITECTURE%_OK"\r');}catch(e){clearTimeout(timer);reject(e);}})();});}
 finally{clearTimeout(timer);if(off)off();await api.dispose(s.id);}});
 assert.deepEqual(errors,[]);await page.screenshot({path:path.join(b.output,'window.png')});
 fs.writeFileSync(path.join(b.output,'acceptance.json'),JSON.stringify({result:'WINDOW_PTY_PASS',binding:b,facts,pty,errors},null,2));
 }finally{if(app)await app.close();}})().catch(e=>{fs.writeFileSync(path.join(b.output,'failure.txt'),e.stack);process.exitCode=1;});

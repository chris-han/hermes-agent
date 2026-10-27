// Package-bound Windows acceptance; no dev renderer or synthetic backend.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { _electron, expect } from '@playwright/test'

const [bindingPath, outputArg] = process.argv.slice(2)
assert.equal(process.platform, 'win32', 'Windows packaged acceptance only')
assert.ok(bindingPath && outputArg, 'usage: node accept-semantier-theme-windows.mjs binding.json NEW_OWNED_OUTPUT')
const binding = JSON.parse(fs.readFileSync(bindingPath, 'utf8'))
assert.match(binding.source, /^[a-f0-9]{40}$/)
assert.match(binding.version, /^\d+\.\d+\.\d+$/)
const digest = file => {
  const hash=crypto.createHash('sha256'), buffer=Buffer.alloc(1024*1024), fd=fs.openSync(file,'r')
  try {
    let size
    while((size=fs.readSync(fd,buffer,0,buffer.length,null))>0) hash.update(buffer.subarray(0,size))
    return hash.digest('hex')
  } finally { fs.closeSync(fd) }
}
for (const name of ['executable', 'asar', 'msix']) {
  assert.ok(path.isAbsolute(binding[name].path))
  assert.equal(digest(binding[name].path), binding[name].sha256, name + ' binding')
}
assert.equal(path.basename(binding.executable.path), 'Semantier Agent.exe')
assert.equal(path.resolve(binding.asar.path), path.resolve(path.dirname(binding.executable.path), 'resources', 'app.asar'))
const output = path.resolve(outputArg)
assert.ok(!fs.existsSync(output), 'output must be fresh and owned')
fs.mkdirSync(output, { recursive: true })
fs.writeFileSync(path.join(output, 'owner.json'), JSON.stringify({ owner: 'windows-semantier-theme-build-v1', source: binding.source }))
const checks = [], errors = [], captures = [], identities = []
let instance
const baseEnv = Object.fromEntries(Object.entries(process.env).filter(([k]) =>
  /^(PATH|SYSTEMROOT|WINDIR|COMSPEC|PATHEXT|USERPROFILE|SYSTEMDRIVE|PROCESSOR_ARCHITECTURE|NUMBER_OF_PROCESSORS)$/i.test(k)))
function environment(root) {
  const dirs = Object.fromEntries(['home','appdata','localappdata','temp','userdata','hermes'].map(name => [name, path.join(root,name)]))
  for (const dir of Object.values(dirs)) fs.mkdirSync(dir,{recursive:true})
  return { ...baseEnv, HOME:dirs.home, APPDATA:dirs.appdata, LOCALAPPDATA:dirs.localappdata, TEMP:dirs.temp, TMP:dirs.temp,
    HERMES_HOME:dirs.hermes, HERMES_DESKTOP_USER_DATA_DIR:dirs.userdata,
    HERMES_DESKTOP_IGNORE_EXISTING:'1', HERMES_DESKTOP_SKIP_QUIT_CONFIRM:'1',
    HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION:'1', HERMES_DESKTOP_CDP_PORT:'off',
    HTTP_PROXY:'http://127.0.0.1:9', HTTPS_PROXY:'http://127.0.0.1:9', ALL_PROXY:'http://127.0.0.1:9',
    NO_PROXY:'127.0.0.1,localhost,::1' }
}
async function close() {
  if (instance) { await instance.app.close(); instance = null }
}
async function launch(root) {
  assert.ok(path.resolve(root).startsWith(output + path.sep))
  const app = await _electron.launch({ executablePath:binding.executable.path, args:[], env:environment(root), timeout:120000 })
  instance = {app}
  const identity = await app.evaluate(({app}) => ({packaged:app.isPackaged, version:app.getVersion(), executable:process.execPath,
    platform:process.platform, resources:process.resourcesPath, userData:app.getPath('userData'), locale:app.getLocale()}))
  assert.equal(identity.platform,'win32'); assert.equal(identity.packaged,true)
  assert.equal(identity.version,binding.version)
  assert.equal(path.resolve(identity.executable),path.resolve(binding.executable.path))
  assert.equal(path.resolve(identity.userData),path.resolve(root,'userdata'))
  const page = await app.firstWindow({timeout:120000})
  await page.context().route('**/*', async route => {
    const url = new URL(route.request().url())
    if (['http:','https:'].includes(url.protocol) && !['localhost','127.0.0.1','[::1]'].includes(url.hostname)) return route.abort()
    return route.continue()
  })
  page.on('pageerror',error=>errors.push(error.message))
  await page.waitForLoadState('domcontentloaded')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme', /.+/, {timeout:120000})
  identities.push({...identity,pid:app.process().pid,produced_at:new Date().toISOString()})
  await page.screenshot({path:path.join(output,'first-window-'+identities.length+'.png')})
  instance = {app,page,identity}
  return instance
}
async function routeSettings(page, subpage) {
  await page.evaluate(value=>{location.hash='/settings?tab=config:appearance&page='+value},subpage)
}
async function language(page, id) {
  await routeSettings(page,'general')
  const trigger=page.getByRole('button',{name:/Switch language|切换语言/i})
  await trigger.click()
  await page.locator('[cmdk-item][data-value="'+id+'"]').click()
  await expect(page.locator('html')).toHaveAttribute('lang',id)
  await expect(page.getByRole('button',{name:/Switch language|切换语言/i})).toBeEnabled()
  await expect(page.locator('[cmdk-input]')).toHaveCount(0)
}
async function mode(page, value) {
  await routeSettings(page,'theme')
  const names={light:/^Light$|^明亮$/,dark:/^Dark$|^暗色$/,system:/^System$|^跟随系统$|^系统$/}
  await page.getByRole('button',{name:names[value]}).click()
  if(value!=='system') await expect(page.locator('html')).toHaveAttribute('data-hermes-mode',value)
}

async function contrastRoles(page) {
  const result=await page.evaluate(()=>{
    const style=getComputedStyle(document.documentElement)
    const canvas=new OffscreenCanvas(1,1), context=canvas.getContext('2d',{willReadFrequently:true})
    const color=value=>{
      if(!value.trim()) throw new Error('missing rendered role')
      context.clearRect(0,0,1,1);context.fillStyle=value;context.fillRect(0,0,1,1)
      return [...context.getImageData(0,0,1,1).data]
    }
    const luminance=rgb=>rgb.slice(0,3).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4)
      .reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0)
    const definitions=[
      ['action','--dt-primary-solid-foreground','--dt-primary-solid',4.5],
      ['intelligence','--ui-intelligence-foreground','--ui-intelligence',4.5],
      ['text','--ui-text-primary','--ui-bg-editor',4.5],
      ['secondary','--ui-text-secondary','--ui-bg-editor',4.5],
      ['tertiary','--ui-text-tertiary','--ui-bg-editor',4.5],
      ['destructive','--dt-destructive-foreground','--dt-destructive',4.5],
      ['focus','--dt-ring','--ui-bg-chrome',3],
      ['success','--ui-success','--ui-bg-editor',4.5],
      ['warning','--ui-warning','--ui-bg-editor',4.5],
      ['info','--ui-info','--ui-bg-editor',4.5]
    ]
    return definitions.map(([name,fg,bg,minimum])=>{
      const foreground=color(style.getPropertyValue(fg)),background=color(style.getPropertyValue(bg))
      const a=luminance(foreground),b=luminance(background)
      return {name,foreground,background,minimum,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)}
    })
  })
  for(const row of result) assert.ok(row.ratio>=row.minimum,JSON.stringify(row))
  return result
}

async function capture(page,app,name,width) {
  await app.evaluate(({BrowserWindow},width)=>BrowserWindow.getAllWindows()[0].setSize(width,1000),width)
  await page.emulateMedia({reducedMotion:'reduce'})
  const bounds=await app.evaluate(({BrowserWindow})=>({bounds:BrowserWindow.getAllWindows()[0].getBounds(),minimum:BrowserWindow.getAllWindows()[0].getMinimumSize()}))
  await page.evaluate(()=>document.fonts.ready)
  const geometry=await page.evaluate(()=>({width:innerWidth,height:innerHeight,scroll:document.documentElement.scrollWidth,
    theme:document.documentElement.dataset.hermesTheme,mode:document.documentElement.dataset.hermesMode,lang:document.documentElement.lang,
    fonts:[...document.fonts].filter(f=>f.status==='loaded').map(f=>({family:f.family,weight:f.weight}))}))
  assert.ok(geometry.scroll<=geometry.width+1,'shell overflow')
  await page.screenshot({path:path.join(output,name+'.png'),fullPage:true})
  captures.push({name,bounds,geometry,contrast:await contrastRoles(page)})
}

const observations = []
async function productionSurfaces(page, app) {
  await language(page, 'en')
  await routeSettings(page, 'theme')
  const search = page.getByPlaceholder(/search.*theme/i)
  await search.fill('native-no-matching-theme-20261027')
  await expect(page.getByText(/No installed themes match/)).toBeVisible()
  await search.fill('')
  const alternate = page.getByRole('button', {name:/Classic Hermes/})
  await alternate.click()
  await expect(page.locator('html')).not.toHaveAttribute('data-hermes-design', 'asimov')
  await page.getByRole('button', {name:/^Semantier/}).click()
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme', 'semantier')
  // Real command palette preview, cancelled without committing.
  await page.evaluate(()=>{location.hash='/'})
  await expect(search).toHaveCount(0)
  await page.keyboard.press('Control+k')
  const input = page.locator('[cmdk-input]')
  await input.fill('Everforest')
  await page.locator('[cmdk-item]').filter({hasText:'Everforest'}).first().hover()
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme', 'everforest')
  await page.keyboard.press('Escape')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme', 'semantier')
  await routeSettings(page,'theme')
  await page.getByRole('button',{name:/Classic Hermes/}).click()
  await page.evaluate(()=>{location.hash='/'})
  const composer=page.locator('[data-slot="composer-rich-input"][contenteditable="true"]').first()
  await composer.fill('/skin default')
  await composer.press('Escape')
  await composer.press('Enter')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme','semantier')
  checks.push('T11 alternate theme, cancelled production palette preview and local /skin default')
  // Measure the actual language popover and its focus lifecycle.
  await routeSettings(page, 'general')
  const trigger = page.getByRole('button', {name:/Switch language/i})
  await trigger.focus()
  await trigger.press('Enter')
  await expect(page.locator('[cmdk-input]')).toBeFocused()
  const surface = await page.locator('[cmdk-root]').evaluate(el => {
    const cs=getComputedStyle(el.closest('[data-radix-popper-content-wrapper]')?.firstElementChild || el)
    return {background:cs.backgroundColor,shadow:cs.boxShadow,radius:cs.borderRadius}
  })
  observations.push({overlay:surface})
  await page.keyboard.press('Escape')
  await expect(trigger).toBeFocused()
  checks.push('T10 production language overlay keyboard entry/Escape/return')
  // System mode follows the native Electron theme service, without changing Windows settings.
  await mode(page,'system')
  for(const theme of ['dark','light']) {
    await app.evaluate(({nativeTheme},value)=>{nativeTheme.themeSource=value},theme)
    await expect(page.locator('html')).toHaveAttribute('data-hermes-mode',theme)
  }
  await app.evaluate(({nativeTheme})=>{nativeTheme.themeSource='system'})
  observations.push({systemMode:'nativeTheme dark/light/system'})
  // Open the actual integrated terminal; no shell or model command is sent.
  await page.evaluate(()=>{location.hash='/'})
  await page.keyboard.press('Control+Backquote')
  const terminal=page.locator('.xterm').first()
  await expect(terminal).toBeVisible({timeout:60000})
  await terminal.locator('textarea').focus()
  await page.keyboard.type('0O1Il') // Visible glyph sample only; never press Enter.
  for(const appearance of ['light','dark']) {
    await mode(page,appearance)
    await page.evaluate(()=>{location.hash='/'})
    await expect(terminal).toBeVisible()
    observations.push({terminal:appearance,style:await terminal.evaluate(el=>{
      const s=getComputedStyle(el);return {font:s.fontFamily,foreground:s.color,background:s.backgroundColor}
    })})
    await capture(page,app,'terminal-'+appearance,1440)
  }
  await terminal.locator('textarea').press('Control+u')
  checks.push('T13 real integrated terminal in light/dark; no command sent')
}

async function createProfile(page,name) {
  await page.evaluate(()=>{location.hash='/profiles'})
  await page.getByRole('button',{name:'New profile',exact:true}).click()
  const dialog=page.getByRole('dialog').filter({has:page.locator('#new-profile-name')})
  await expect(dialog.getByRole('button',{name:'Create profile',exact:true})).toBeDisabled()
  await expect(dialog.locator('#new-profile-name')).toBeFocused()
  if(name==='legacy') {
    await dialog.locator('#new-profile-name').fill('x'.repeat(65))
    await expect(dialog.locator('#new-profile-name')).toHaveAttribute('aria-invalid','true')
    await expect(dialog.getByRole('button',{name:'Create profile',exact:true})).toBeDisabled()
    observations.push({invalidProfile:await dialog.evaluate(el=>{
      const s=getComputedStyle(el);return {background:s.backgroundColor,shadow:s.boxShadow,radius:s.borderRadius}
    })})
    await page.screenshot({path:path.join(output,'profile-invalid-and-disabled.png'),fullPage:true})
    await dialog.locator('#new-profile-clone-from').click()
    await expect(page.getByRole('listbox')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(dialog.locator('#new-profile-clone-from')).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(dialog).toHaveCount(0)
    await page.getByRole('button',{name:'New profile',exact:true}).click()
  }
  await dialog.locator('#new-profile-name').fill(name)
  await dialog.locator('#new-profile-soul').fill('native-theme-preservation:'+name)
  await dialog.getByRole('button',{name:'Create profile',exact:true}).click()
  await expect(dialog).toHaveCount(0,{timeout:60000})
  observations.push({createdProfile:name,via:'production profile dialog'})
}
async function selectProfile(page,name) {
  await page.evaluate(()=>{location.hash='/'})
  const composer=page.locator('[data-slot="composer-rich-input"][contenteditable="true"]').first()
  await composer.fill('/profile '+name)
  await composer.press('Escape')
  await composer.press('Enter')
  await expect.poll(()=>page.evaluate(()=>localStorage.getItem('hermes-desktop-active-profile-v1')),{timeout:120000}).toBe(name)
}

async function fontsAndRoles(page) {
  const evidence=await page.evaluate(async()=>{
    const samples=[['Manrope','Semantier'],['Noto Sans SC','中文默认语言'],['JetBrains Mono','0O1Il']]
    const fonts=[]
    for(const [family,sample] of samples) for(const weight of [400,500,600,700]) {
      const loaded=await document.fonts.load(weight+' 16px "'+family+'"',sample)
      fonts.push({family,weight,count:loaded.length,status:loaded.map(f=>f.status),matched:loaded.map(f=>({family:f.family,weight:f.weight,style:f.style}))})
    }
    const style=getComputedStyle(document.documentElement)
    const names=['--theme-background-seed','--theme-foreground','--theme-primary','--dt-primary-foreground','--dt-destructive','--dt-ring',
      '--ui-bg-chrome','--ui-text-primary','--ui-text-secondary','--dt-primary-solid','--ui-intelligence']
    return {fonts,roles:Object.fromEntries(names.map(n=>[n,style.getPropertyValue(n)])),
      controls:[...document.querySelectorAll('button,input,[role="combobox"]')].filter(e=>e.getBoundingClientRect().width>0).slice(0,60).map(el=>{
        const c=getComputedStyle(el),r=el.getBoundingClientRect()
        return {tag:el.tagName,label:el.getAttribute('aria-label')||el.textContent?.slice(0,60),disabled:el.disabled,
          foreground:c.color,background:c.backgroundColor,border:c.borderColor,radius:c.borderRadius,
          shadow:c.boxShadow,font:c.fontFamily,width:r.width,height:r.height}
      })}
  })
  assert.ok(evidence.fonts.every(f=>f.count>0&&f.status.every(s=>s==='loaded')),'bundled required font faces')
  observations.push(evidence)
  checks.push('T07 required bundled font faces loaded offline; glyph screenshots retained')
}

try {
  const fresh=path.join(output,'fresh')
  let {app,page,identity}=await launch(fresh)
  assert.match(identity.locale,/^en/i,'fresh Chinese case requires actual English host app locale')
  await expect(page.locator('html')).toHaveAttribute('lang','zh',{timeout:120000})
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme','semantier')
  checks.push('T02 packaged Windows identity and isolated userData','T03 fresh Chinese and Semantier on English host')
  for(const lang of ['en','zh']) {
    await language(page,lang)
    for(const appearance of ['light','dark']) {
      await mode(page,appearance)
      for(const width of [1440,760]) await capture(page,app,lang+'-'+appearance+'-'+width,width)
    }
  }
  checks.push('T06 EN/ZH light/dark desktop and actual narrow bounds')
  await fontsAndRoles(page)
  await productionSurfaces(page,app)
  await language(page,'en')
  await page.evaluate(()=>localStorage.setItem('native-theme-nonappearance-sentinel','retain-me'))
  await close()
  ;({app,page}=await launch(fresh))
  await expect(page.locator('html')).toHaveAttribute('lang','en',{timeout:120000})
  assert.equal(await page.evaluate(()=>localStorage.getItem('native-theme-nonappearance-sentinel')),'retain-me')
  await language(page,'zh')
  await close()
  ;({app,page}=await launch(fresh))
  await expect(page.locator('html')).toHaveAttribute('lang','zh',{timeout:120000})
  checks.push('T12 explicit English and Chinese survive full process restart; nonlanguage sentinel retained')
  await language(page,'en')
  for(const name of ['legacy','explicit','inherited']) await createProfile(page,name)
  const souls=Object.fromEntries(['legacy','explicit','inherited'].map(name=>{
    const file=path.join(fresh,'hermes','profiles',name,'SOUL.md')
    assert.ok(fs.readFileSync(file,'utf8').includes('native-theme-preservation:'+name))
    return [file,digest(file)]
  }))
  await page.evaluate(()=>{
    localStorage.setItem('hermes-desktop-theme-v2','nous-alt')
    localStorage.setItem('hermes-desktop-mode-v1','light')
    localStorage.setItem('hermes-desktop-profile-themes-v1',JSON.stringify({legacy:'nous-alt',explicit:'nous'}))
    localStorage.setItem('hermes-desktop-profile-modes-v1',JSON.stringify({legacy:'dark',explicit:'system'}))
  })
  await close()
  ;({app,page}=await launch(fresh))
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme','semantier')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-mode','light')
  assert.equal(await page.evaluate(()=>localStorage.getItem('native-theme-nonappearance-sentinel')),'retain-me')
  checks.push('T04 legacy global Nous Alt migration and saved light mode')
  const migrated=await page.evaluate(()=>({
    themes:JSON.parse(localStorage.getItem('hermes-desktop-profile-themes-v1')),
    modes:JSON.parse(localStorage.getItem('hermes-desktop-profile-modes-v1'))
  }))
  assert.deepEqual(migrated.themes,{legacy:'semantier',explicit:'nous'})
  assert.deepEqual(migrated.modes,{legacy:'dark',explicit:'system'})
  observations.push({mixedProfileMigration:migrated})
  await selectProfile(page,'legacy')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme','semantier')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-mode','dark')
  await language(page,'zh')
  await selectProfile(page,'explicit')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme','nous')
  await expect(page.locator('html')).toHaveAttribute('lang','en',{timeout:60000})
  await selectProfile(page,'inherited')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme','semantier')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-mode','light')
  await expect(page.locator('html')).toHaveAttribute('lang','en',{timeout:60000})
  await selectProfile(page,'legacy')
  await expect(page.locator('html')).toHaveAttribute('lang','zh',{timeout:60000})
  await selectProfile(page,'default')
  checks.push('T05 real profiles migrate/retain/inherit appearance; profile language choices remain separate')
  await language(page,'en')
  await routeSettings(page,'theme')
  await page.getByRole('button',{name:/Classic Hermes/}).click()
  await mode(page,'dark')
  await close()
  ;({app,page}=await launch(fresh))
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme','classic')
  await expect(page.locator('html')).toHaveAttribute('data-hermes-mode','dark')
  await expect(page.locator('html')).toHaveAttribute('lang','en')
  for(const [file,sha] of Object.entries(souls)) assert.equal(digest(file),sha,'profile SOUL preservation')
  observations.push({nonappearanceSoulHashes:souls})
  checks.push('T12 explicit alternate theme/dark/English and profile SOUL data survive full restart')

  await close()
  assert.deepEqual(errors,[])
} catch(error) {
  errors.push(String(error.stack||error))
  throw error
} finally {
  await close()
  fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify({produced_at:new Date().toISOString(),binding,harness_sha256:digest(fileURLToPath(import.meta.url)),checks,errors,captures,observations,identities,
    pending:['T05 profile inheritance','T07 offline glyph inspection','T08 contrast','T09 controls','T10 overlays','T11 preset/preview/command','T13 terminal','T14 integration'],
    registered_installation:'NOT_RUN',human_usability:'UNMEASURED'},null,2))
}

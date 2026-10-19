// Real production Electron renderer/preload/main, isolated from normal app data.
// Only backend bootstrap is deliberately unavailable; branding needs no backend.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { _electron, expect } from '@playwright/test'
const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(desktop, 'package.json'))
const output = process.argv[2] ?? fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-branding-ui-'))
fs.mkdirSync(output,{recursive:true})
const data = path.join(output,'user-data')
const env = {...process.env,HERMES_HOME:path.join(output,'hermes-home'),HERMES_DESKTOP_USER_DATA_DIR:data,HERMES_DESKTOP_IGNORE_EXISTING:'1',HERMES_DESKTOP_SKIP_QUIT_CONFIRM:'1',HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION:'1',HERMES_DESKTOP_CDP_PORT:'off',HERMES_DESKTOP_BOOT_FAKE_ERROR:'Offline branding acceptance: backend not required'}
const checks=[]
async function launch() {
 const app=await _electron.launch({executablePath:require('electron'),args:[desktop,'--disable-gpu'],cwd:desktop,env})
 const page=await app.firstWindow()
 await page.waitForLoadState('domcontentloaded')
 await page.evaluate(()=>{location.hash='/settings?tab=config:appearance&page=customize'})
 await expect(page.getByLabel('Organization display name (English)')).toBeVisible({timeout:30000})
 // The production failure card may appear after route mount; Escape dismisses it.
 await page.waitForTimeout(700);await page.keyboard.press('Escape')
 const dismiss=page.getByRole('button',{name:'Dismiss',exact:true});if(await dismiss.count()) await dismiss.first().click()
 return {app,page}
}
let instance
try {
 instance=await launch();let {app,page}=instance
 await page.getByLabel('Organization display name (English)').fill('Native Brand')
 await page.getByLabel('Organization display name (Chinese)').fill('原生品牌')
 const png=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=160;c.height=64;const x=c.getContext('2d');x.fillStyle='#406080';x.fillRect(8,8,144,48);return c.toDataURL('image/png').split(',')[1]})
 await page.getByLabel('Organization logo',{exact:true}).setInputFiles({name:'wide.png',mimeType:'image/png',buffer:Buffer.from(png,'base64')})
 await expect(page.getByRole('region',{name:'Placement preview'}).locator('img')).toBeVisible()
 assert.equal((await page.evaluate(()=>window.hermesDesktop.branding.get())).displayName,'')
 const placement=page.getByRole('navigation',{name:'Placements'}).getByRole('button')
 assert.equal(await placement.count(),8)
 for(let i=0;i<8;i++){await placement.nth(i).click();await expect(page.getByRole('region',{name:'Placement preview'})).toContainText('Native Brand')}
 checks.push('all eight draft previews; real PNG decode; production state unchanged before Apply')
 await page.getByLabel('Organization logo',{exact:true}).setInputFiles({name:'spoof.png',mimeType:'image/png',buffer:Buffer.from('<svg/>')})
 await expect(page.getByRole('alert').filter({hasText:'Choose a valid'})).toContainText('Choose a valid')
 await expect(page.getByRole('region',{name:'Placement preview'}).locator('img')).toBeVisible()
 checks.push('invalid upload retains prior raster')
 for(const format of ['jpeg','webp']) {
  const encoded=await page.evaluate(format=>{const c=document.createElement('canvas');c.width=160;c.height=64;const x=c.getContext('2d');x.fillRect(8,8,144,48);return c.toDataURL('image/'+format).split(',')[1]},format)
  await page.getByLabel('Organization logo',{exact:true}).setInputFiles({name:'wide.'+format,mimeType:'image/'+format,buffer:Buffer.from(encoded,'base64')})
  await expect(page.getByRole('region',{name:'Placement preview'}).locator('img')).toHaveJSProperty('naturalWidth',160)
  await expect(page.getByRole('region',{name:'Placement preview'}).locator('img')).toHaveJSProperty('naturalHeight',64)
 }
 for(const buffer of [Buffer.alloc(2*1024*1024+1),Buffer.from(await page.evaluate(()=>{const c=document.createElement('canvas');c.width=16;c.height=16;return c.toDataURL('image/png').split(',')[1]}),'base64')]) {
  await page.getByLabel('Organization logo',{exact:true}).setInputFiles({name:'invalid.png',mimeType:'image/png',buffer})
  await expect(page.getByRole('alert').filter({hasText:'Choose a valid'})).toBeVisible()
 }
 checks.push('real JPEG/WebP decode preserves aspect ratio; oversized and invalid-dimension images rejected')
 const waiting=app.waitForEvent('window');await page.evaluate(async()=>{await window.hermesDesktop.openWindow();return true});const peer=await waiting;await peer.waitForLoadState('domcontentloaded')
 fs.mkdirSync(path.join(data,'desktop-branding.v1.json.tmp'))
 await page.getByRole('button',{name:'Apply changes',exact:true}).click()
 await expect(page.getByRole('alert').filter({hasText:'Could not save'})).toContainText('Could not save')
 assert.equal((await page.evaluate(()=>window.hermesDesktop.branding.get())).displayName,'')
 fs.rmdirSync(path.join(data,'desktop-branding.v1.json.tmp'))
 await page.getByRole('button',{name:'Apply changes',exact:true}).click()
 await expect(page.getByText('Changes saved',{exact:true})).toBeVisible()
 await expect(peer.locator('[data-organization-brand]').first()).toContainText('Native Brand')
 checks.push('failed native write retains draft and committed state; retry; actual peer renderer synchronizes')
 await page.getByLabel('Organization display name (English)').scrollIntoViewIfNeeded()
 await page.screenshot({path:path.join(output,'native-customize.png'),fullPage:true})
 await app.evaluate(({BrowserWindow})=>{const win=BrowserWindow.getAllWindows()[0];win.setSize(900,760)})
 await page.getByRole('button',{name:'Apply changes',exact:true}).scrollIntoViewIfNeeded()
 await page.screenshot({path:path.join(output,'native-compact-actions.png'),fullPage:true})
 await app.evaluate(({BrowserWindow})=>{const win=BrowserWindow.getAllWindows()[0];win.setSize(1280,860)})
 await page.evaluate(()=>{location.hash='/settings?tab=about&page=updates'})
 await expect(page.locator('[data-organization-brand]').filter({hasText:'Native Brand'}).last()).toBeVisible()
 await page.screenshot({path:path.join(output,'native-about.png'),fullPage:true})
 await app.close();instance=null
 instance=await launch();({app,page}=instance)
 const restored=await page.evaluate(()=>window.hermesDesktop.branding.get());assert.equal(restored.displayName,'Native Brand');assert.equal(restored.displayNameZh,'原生品牌');assert.ok(restored.logo)
 await expect(page.getByLabel('Organization display name (English)')).toHaveValue('Native Brand')
 checks.push('complete production Electron restart restores name and logo before settings')
 await page.getByRole('button',{name:'Remove logo',exact:true}).click();await expect(page.getByRole('region',{name:'Placement preview'}).locator('img')).toHaveAttribute('src',/semantier-logo/);await page.getByRole('button',{name:'Apply changes',exact:true}).click();await expect(page.getByText('Changes saved',{exact:true})).toBeVisible()
 await app.close();instance=null
 instance=await launch();({app,page}=instance)
 assert.deepEqual(await page.evaluate(()=>window.hermesDesktop.branding.get()),{version:1,displayName:'Native Brand',displayNameZh:'原生品牌',logo:null})
 await page.getByRole('button',{name:'Restore defaults',exact:true}).click();await page.getByRole('button',{name:'Apply changes',exact:true}).click();await expect(page.getByText('Changes saved',{exact:true})).toBeVisible()
 await app.close();instance=null
 instance=await launch();({app,page}=instance)
 assert.deepEqual(await page.evaluate(()=>window.hermesDesktop.branding.get()),{version:1,displayName:'',displayNameZh:'',logo:null})
 await expect(page.getByRole('region',{name:'Placement preview'})).toContainText('Semantier')
 await expect(page.getByRole('region',{name:'Placement preview'}).locator('img')).toHaveAttribute('src',/semantier-logo/)
 checks.push('remove preserves both names across full restart; reset restores defaults across full restart')
 fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify({producedAt:new Date().toISOString(),platform:process.platform,electron:await app.evaluate(()=>process.versions.electron),checks,limitations:['Backend bootstrap intentionally unavailable; no backend/auth/provider/model acceptance claimed','Chinese and compact/theme rendering checked separately']},null,2))
 console.log(JSON.stringify({ok:true,checks,output}))
} finally {if(instance)await instance.app.close();fs.rmSync(data,{recursive:true,force:true});fs.rmSync(path.join(output,'hermes-home'),{recursive:true,force:true})}

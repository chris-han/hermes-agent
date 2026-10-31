/** First-party Semantier journey producer. Explicit consent; no identity or business-fact claims. */
import fs from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'

type LaunchRecord = { installation_id:string; event_id:string; occurred_at:string; acknowledged:boolean }
export class JourneyObservations {
  private record:LaunchRecord|null=null
  private file:string
  private running:Promise<void>|null=null
  private failed=false
  private stateQueue:Promise<void>=Promise.resolve()
  private queuedStates=0
  private unavailable=false
  private originAvailable=false
  constructor(private dir:string, private origin:string, private enabled:()=>boolean, private send:typeof fetch=fetch) {
    this.file=path.join(dir,'semantier-journey-first-open.json')
    try {
      const url=new URL(origin)
      this.originAvailable=!(url.username || url.password || url.search || url.hash || url.pathname!=='/' || !(url.protocol==='https:' || url.protocol==='http:' && ['localhost','127.0.0.1'].includes(url.hostname)))
    } catch {this.originAvailable=false}
    try {
      if (fs.existsSync(this.file)) {
        const info=fs.lstatSync(this.file)
        if (info.isSymbolicLink() || info.size>2048) throw Error('Invalid optional observation')
        const value=JSON.parse(fs.readFileSync(this.file,'utf8'))
        if (!/^[a-f0-9-]{36}$/.test(value.installation_id) || !/^[a-f0-9-]{36}$/.test(value.event_id) || !Number.isFinite(Date.parse(value.occurred_at)) || typeof value.acknowledged!=='boolean') throw Error('Invalid optional observation')
        this.record=value
      }
    } catch {this.unavailable=true}
  }
  private save() {
    const temporary=this.file+'.'+randomUUID()+'.tmp'
    try {
      fs.mkdirSync(this.dir,{recursive:true})
      fs.writeFileSync(temporary,JSON.stringify(this.record),{flag:'wx',mode:0o600})
      fs.renameSync(temporary,this.file)
    } finally {try{fs.unlinkSync(temporary)}catch{/* already renamed */}}
  }
  status() {return {enabled:this.enabled(),originAvailable:this.originAvailable,storageUnavailable:this.unavailable,acknowledged:this.record?.acknowledged ?? false}}
  /** Invoked by the real app.whenReady lifecycle; optional transport never gates startup. */
  firstOpen():Promise<void> {
    if (this.running) return this.running
    if (!this.record && !this.unavailable) {
      try {this.record={installation_id:randomUUID(),event_id:randomUUID(),occurred_at:new Date().toISOString(),acknowledged:false};this.save()}
      catch {this.unavailable=true}
    }
    this.running=this.deliver().finally(()=>{this.running=null})
    return this.running
  }
  /** Main-frame load failures/recovery only; no URL, error prose or customer content. */
  rendererState(failed:boolean):Promise<void> {
    if(this.queuedStates>=8)return Promise.resolve()
    const occurred_at=new Date().toISOString()
    this.queuedStates++
    this.stateQueue=this.stateQueue.then(async()=>{
      if(!this.enabled() || !this.originAvailable || this.unavailable || !this.record || (!failed&&!this.failed))return
      try {
        const session=await this.send(this.origin+'/api/journey/session',{method:'POST',credentials:'include',signal:AbortSignal.timeout(3000)})
        if(!session.ok)return
        const response=await this.send(this.origin+'/api/journey/observations/windows',{method:'POST',credentials:'include',headers:{'content-type':'application/json'},body:JSON.stringify({event_id:randomUUID(),event_name:failed?'client_failed':'client_resumed',occurred_at,installation_id:this.record.installation_id,producer_version:'electron-journey-v1',properties:{reason_code:failed?'renderer_load_failed':'renderer_load_recovered'}}),signal:AbortSignal.timeout(3000)})
        if(response.ok)this.failed=failed
      } catch {/* Optional telemetry cannot block renderer recovery. */}
    }).finally(()=>{this.queuedStates--})
    return this.stateQueue
  }
  private async deliver() {
    if (!this.enabled() || !this.originAvailable || this.unavailable || this.record?.acknowledged) return
    try {
      if (!this.record) {this.record={installation_id:randomUUID(),event_id:randomUUID(),occurred_at:new Date().toISOString(),acknowledged:false};this.save()}
      // One retained observation, at most three attempts per launch. No disk secret or customer content.
      for (let attempt=0;attempt<3 && this.enabled();attempt++) {
        try {
          const session=await this.send(this.origin+'/api/journey/session',{method:'POST',credentials:'include',signal:AbortSignal.timeout(3000)})
          if (!session.ok) continue
          const r=await this.send(this.origin+'/api/journey/observations/windows',{method:'POST',credentials:'include',headers:{'content-type':'application/json'},body:JSON.stringify({event_id:this.record.event_id,event_name:'app_first_open',occurred_at:this.record.occurred_at,installation_id:this.record.installation_id,producer_version:'electron-journey-v1',properties:{}}),signal:AbortSignal.timeout(3000)})
          if (r.ok) {this.record.acknowledged=true;this.save();break}
          if (r.status>=400 && r.status<500) break
        } catch {/* bounded, optional collection failure */}
      }
    } catch {this.unavailable=true}
  }
}

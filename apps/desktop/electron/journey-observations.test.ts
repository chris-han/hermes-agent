import { mkdtempSync,readFileSync,rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach,expect,it,vi } from 'vitest'
import { JourneyObservations } from './journey-observations'
const dirs:string[]=[]
function root(){const d=mkdtempSync(path.join(os.tmpdir(),'semantier-journey-'));dirs.push(d);return d}
afterEach(()=>{for(const d of dirs.splice(0))rmSync(d,{recursive:true,force:true})})
it('sends nothing without explicit consent',async()=>{
 const send=vi.fn();const owner=new JourneyObservations(root(),'http://127.0.0.1:8899',()=>false,send)
 await owner.firstOpen();expect(send).not.toHaveBeenCalled()
})
it('emits exactly one first-open across repeated launches with stable installation and event identity',async()=>{
 const dir=root();const send=vi.fn().mockImplementation(async()=>Response.json({status:'accepted'}))
 const owner=new JourneyObservations(dir,'http://127.0.0.1:8899',()=>true,send)
 await Promise.all([owner.firstOpen(),owner.firstOpen()]);expect(send).toHaveBeenCalledTimes(2)
 const body=JSON.parse(send.mock.calls[1][1].body);expect(body.event_name).toBe('app_first_open');expect(body.user_id).toBeUndefined()
 await new JourneyObservations(dir,'http://127.0.0.1:8899',()=>true,send).firstOpen();expect(send).toHaveBeenCalledTimes(2)
 expect(JSON.parse(readFileSync(path.join(dir,'semantier-journey-first-open.json'),'utf8')).acknowledged).toBe(true)
})
it('bounds offline retries without blocking startup or fabricating acknowledgement',async()=>{
 const send=vi.fn().mockRejectedValue(Error('offline'));const owner=new JourneyObservations(root(),'http://127.0.0.1:8899',()=>true,send)
 await owner.firstOpen();expect(send).toHaveBeenCalledTimes(3);expect(owner.status().acknowledged).toBe(false)
})
it('records a renderer failure before recovery without retaining error prose',async()=>{
 const send=vi.fn().mockImplementation(async()=>Response.json({status:'accepted'}))
 const owner=new JourneyObservations(root(),'http://127.0.0.1:8899',()=>true,send)
 await owner.firstOpen();send.mockClear()
 await owner.rendererState(false);expect(send).not.toHaveBeenCalled()
 await Promise.all([owner.rendererState(true),owner.rendererState(false)])
 const bodies=send.mock.calls.filter(([,init])=>init.body).map(([,init])=>JSON.parse(init.body))
 expect(bodies.map(b=>b.event_name)).toEqual(['client_failed','client_resumed'])
 expect(bodies[0].installation_id).toBe(bodies[1].installation_id)
 expect(bodies[0].properties).toEqual({reason_code:'renderer_load_failed'})
})

import assert from 'node:assert/strict'
import { test } from 'vitest'

import { accessRoute, declineProfileConsent, scoreTrial, unwrapTools } from '../e2e/autonomous-evaluator.ts'
const direct = [{function:{name:'drive_preview',arguments:'{"action":"click"}'}}]
const deferred = [{function:{name:'tool_search',arguments:'{"query":"visible page"}'}},{function:{name:'tool_call',arguments:JSON.stringify({name:'drive_preview',arguments:{action:'click'}})}}]
const base = {requestReached:true,inferenceCalls:2,calls:direct,target:'desktop',ownerCorrect:true,operationSuccess:true,effect:true}
test('1 deferred native discovery and wrapped action accepted',()=>{assert.equal(accessRoute(['tool_search','tool_describe','tool_call'],'desktop_preview drive_preview'),'DEFERRED');assert.equal(scoreTrial({...base,calls:deferred}).workflow,true);assert.equal(unwrapTools(deferred)[1].wrapper,'tool_call')})
test('2 materialized direct action accepted',()=>{assert.equal(accessRoute(['desktop_preview','drive_preview'],''),'DIRECT');assert.equal(scoreTrial(base).workflow,true)})
test('3 no matching authorized tool is explicit unavailable apparatus',()=>assert.equal(scoreTrial({...base,authorizedToolUnavailable:true}).reason,'NO_AUTHORIZED_TOOL'))
test('4 wrong browser cannot pass',()=>assert.equal(scoreTrial({...base,calls:[{name:'browser_exec',arguments:{}}]}).workflow,false))
test('5 correct route failed operation is workflow failure',()=>{const s=scoreTrial({...base,operationSuccess:false});assert.equal(s.routing,true);assert.equal(s.reason,'OPERATION_FAILED')})
test('6 correct operation wrong window fails ownership',()=>assert.equal(scoreTrial({...base,ownerCorrect:false}).reason,'WRONG_OWNER'))
test('7 correct sequence without effect fails',()=>assert.equal(scoreTrial({...base,effect:false}).reason,'NO_VERIFIED_EFFECT'))
test('8 consent blocked is invalid not model failure',()=>assert.equal(scoreTrial({...base,consentBlocked:true}).validity,'INVALID_TEST'))
test('9 supported consent dismissed leaves valid case',()=>assert.equal(scoreTrial({...base,consentBlocked:false}).validity,'VALID'))
test('10 no actual model invocation is invalid',()=>assert.equal(scoreTrial({...base,inferenceCalls:0}).reason,'NO_MODEL_REQUEST'))
test('batch wrappers preserve ordered actual actions',()=>assert.deepEqual(unwrapTools([{name:'tool_call',arguments:{calls:[{name:'desktop_preview',arguments:{action:'read'}},{name:'drive_preview',arguments:{action:'type'}}]}}]).map(x=>x.name),['desktop_preview','drive_preview']))
test('discovery alone never completes a workflow',()=>assert.equal(scoreTrial({...base,calls:deferred.slice(0,1)}).workflow,false))
test('false success and unexpected switch fail',()=>{assert.equal(scoreTrial({...base,falseSuccess:true}).workflow,false);assert.equal(scoreTrial({...base,unexpectedSwitch:true}).workflow,false)})

for (const label of ['Not now', '暂不']) {
  for (const phase of ['initial', 'retry', 'restart', 'multiple mounts']) {
    test(`supported consent ${label}: ${phase}`, async()=>{
      let visible=true, copied=false
      const button={isVisible:async()=>visible,click:async()=>{visible=false},waitFor:async()=>assert.equal(visible,false)}
      const composer={click:async()=>{if(visible){throw Error('consent blocked')}}}

      const page={getByRole:(_r,o)=>{assert.ok(o.name.test(label));

return {first:()=>button}},locator:()=>({locator:()=>({first:()=>composer})})}

      const result=await declineProfileConsent(page,5000)
      assert.equal(result.composer_interactable,true);assert.equal(result.dismissed,1);assert.equal(copied,false)
    })
  }
}

test('native JSON-string batch envelope is unwrapped',()=>assert.equal(unwrapTools([{name:'tool_call',arguments:{calls:JSON.stringify([{name:'desktop_preview',arguments:'{"action":"read"}'}])}}])[0].name,'desktop_preview'))

test('background maintenance is not a browser trial final action',()=>{
 const calls=[{name:'browser_exec',arguments:{}},{name:'skills_list',arguments:{}}]
 const s=scoreTrial({...base,target:'independent',calls})
 assert.equal(s.routing,true)
 assert.equal(unwrapTools(calls)[1].name,'skills_list')
})

"""Offline observer regression tests. Synthetic provider bytes, no network/inference."""
import importlib.util,json,os,pathlib,tempfile,unittest
source=pathlib.Path(__file__).with_name('autonomous-provider.py')
with tempfile.TemporaryDirectory(prefix='semantier-evaluator-unit-') as home:
 p=pathlib.Path(home);env=p/'fixture.env';env.write_text('DASHSCOPE_API_KEY=synthetic-unit-key\nDASHSCOPE_BASE_URL=https://127.0.0.1/v1\n')
 os.environ['DASHSCOPE_API_KEY']='synthetic-unit-key';os.environ['DASHSCOPE_BASE_URL']='https://127.0.0.1/v1';os.environ['SEMANTIER_EVAL_OUTPUT']=home;os.environ['SEMANTIER_EVAL_CREDENTIALS_ENV']=str(env)
 spec=importlib.util.spec_from_file_location('observer',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 class ObserverTests(unittest.TestCase):
  def test_raw_contract_uses_actual_request_schema_and_never_repairs(self):
   schema={'type':'object','properties':{'calls':{'type':'array','items':{'type':'object','required':['name','arguments']}}},'required':['calls']}
   request={'tools':[{'function':{'name':'tool_call','parameters':schema}}]}
   good={'message':{'tool_calls':[{'id':'1','function':{'name':'tool_call','arguments':json.dumps({'calls':[{'name':'drive_preview','arguments':{'action':'type','text':'Armed'}}]})}}]}}
   self.assertEqual(m.raw_contract_status(request,good)[0]['status'],'PASS')
   bad={'message':{'tool_calls':[{'id':'2','function':{'name':'tool_call','arguments':'{"name":"drive_preview","arguments":{"action":"type"}}'}}]}}
   original=json.dumps(bad,sort_keys=True)
   self.assertEqual(m.raw_contract_status(request,bad)[0]['status'],'FAIL')
   self.assertEqual(json.dumps(bad,sort_keys=True),original)
  def test_raw_contract_missing_schema_invalid_json_unknown_tool_and_no_calls(self):
   request={'tools':[{'function':{'name':'drive_preview','parameters':{'type':'object','properties':{'action':{'enum':['elements','type']}},'required':['action']}}}]}
   for args in ('{"action":"fill"}','{'):
    reply={'message':{'tool_calls':[{'function':{'name':'drive_preview','arguments':args}}]}}
    self.assertEqual(m.raw_contract_status(request,reply)[0]['status'],'FAIL')
   reply={'message':{'tool_calls':[{'function':{'name':'foreign_tool','arguments':'{}'}}]}}
   self.assertEqual(m.raw_contract_status(request,reply)[0]['status'],'FAIL')
   self.assertEqual(m.raw_contract_status(request,{'message':{'content':'done'}}),[])
   self.assertEqual(m.raw_contract_status({'tools':[{'function':{'name':'drive_preview'}}]}, {'message':{'tool_calls':[{'function':{'name':'drive_preview','arguments':'{}'}}]}})[0]['status'],'UNPROVEN')
  def test_zero_inference_preflight_reaches_main_composition(self):
   from unittest.mock import patch
   from types import SimpleNamespace
   handler=SimpleNamespace(reply=lambda code,value:(code,value))
   body={'messages':[{'role':'system','content':m.INTENT['prompt']}],'tools':[m.INTENT['schema']]}
   with patch.dict(os.environ,{'SEMANTIER_EVAL_PREFLIGHT_ONLY':'1'}),patch.dict(m.STATE,{'case':'A','calls':[],'counts':{},'composition':[],'failures':[],'preflight_requests':0}),patch.object(m.urllib.request,'urlopen',side_effect=AssertionError('No upstream request during preflight')):
    code,value=m.Handler.forward(handler,body)
    self.assertEqual(code,200)
    self.assertEqual(value['choices'][0]['message']['tool_calls'][0]['function']['name'],'browser_target_intent')
    self.assertEqual(m.STATE['calls'],[])
    main={'messages':[{'role':'system','content':m.HINT}], 'tools':[{'function':{'name':name,'description':'desktop_preview drive_preview'}} for name in ('tool_search','tool_describe','tool_call','browser_exec')]}
    code,value=m.Handler.forward(handler,main)
    self.assertEqual(code,503)
    self.assertEqual(m.STATE['preflight_requests'],2)
    self.assertEqual(m.STATE['failures'],[])

  def test_native_intent_composition_is_distinct_from_main_agent(self):
   body={'messages':[{'role':'system','content':m.INTENT['prompt']}],'tools':[m.INTENT['schema']]}
   self.assertEqual(m.request_composition(body)['access_route'],'NATIVE_TARGET_INTENT')
   self.assertTrue(m.request_composition(body)['valid'])
   body['tools'][0]={'type':'function','function':{'name':'browser_target_intent'}}
   self.assertFalse(m.request_composition(body)['valid'])
  def test_intent_is_allowed_only_for_native_intent_request(self):
   calls=[{'function':{'name':'browser_target_intent','arguments':'{"target_kind":"VISIBLE_PANE","operation":"existing"}'}}]
   self.assertIsNone(m.forbidden_tool(calls,allow_intent=True))
   self.assertIn('outside_authorized',m.forbidden_tool(calls))
  def test_budget_cap_never_expands_native_classifier_limit(self):
   self.assertEqual(m.capped_output({'max_tokens':512})['max_tokens'],512)
   self.assertEqual(m.capped_output({'max_tokens':9000})['max_tokens'],4096)
  def test_renderer_reasoning_parts_are_redacted(self):
   self.assertNotIn('private chain',json.dumps(m.scrub({'parts':[{'type':'reasoning','text':'private chain'}]})))
  def test_single_wrapper_and_discovery_allowed(self):
   self.assertIsNone(m.forbidden_tool([{'function':{'name':'tool_search','arguments':'{"query":"visible page"}'}}]))
   self.assertIsNone(m.forbidden_tool([{'function':{'name':'tool_call','arguments':json.dumps({'name':'desktop_preview','arguments':{'action':'read'}})}}]))
  def test_read_only_procedural_lookup_is_not_a_forbidden_action(self):
   for name in ('skills_list','skill_view'):
    self.assertIsNone(m.forbidden_tool([{'function':{'name':name,'arguments':'{"name":"browser"}'}}]))
  def test_nested_forbidden_action_rejected(self):
   self.assertIn('outside_authorized',m.forbidden_tool([{'function':{'name':'tool_call','arguments':json.dumps({'calls':[{'name':'terminal','arguments':{}}]})}}]))
  def test_external_url_rejected(self):
   self.assertEqual(m.forbidden_tool([{'function':{'name':'desktop_preview','arguments':'{"url":"https://example.com"}'}}]),'external_website_attempt')
  def test_json_usage_and_tools(self):
   raw=json.dumps({'model':'qwen3.5-plus','choices':[{'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}],'usage':{'total_tokens':7}}).encode();r=m.normalize_response(raw,'application/json');self.assertEqual(r['usage']['total_tokens'],7)
  def test_streamed_tool_args_and_usage(self):
   parts=[{'model':'qwen3.5-plus','choices':[{'delta':{'tool_calls':[{'index':0,'id':'call1','function':{'name':'tool_call','arguments':'{"name":'}}]}}]},{'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"drive_preview","arguments":{}}'}}]},'finish_reason':'tool_calls'}]},{'choices':[],'usage':{'prompt_tokens':11,'completion_tokens':5,'total_tokens':16}}]
   raw=('\n'.join('data: '+json.dumps(x) for x in parts)+'\ndata: [DONE]\n').encode();r=m.normalize_response(raw,'text/event-stream');self.assertEqual(r['usage']['total_tokens'],16);self.assertEqual(json.loads(r['message']['tool_calls'][0]['function']['arguments'])['name'],'drive_preview')
  def test_redaction_does_not_archive_key_or_reasoning(self):
   r=m.scrub({'x':m.KEY,'reasoning_content':'private reasoning'});self.assertNotIn(m.KEY,json.dumps(r));self.assertNotIn('private reasoning',json.dumps(r))
 unittest.main(verbosity=2)

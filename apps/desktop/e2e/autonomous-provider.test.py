"""Offline observer regression tests. Synthetic provider bytes, no network/inference."""
import importlib.util,json,os,pathlib,tempfile,unittest
source=pathlib.Path(__file__).with_name('autonomous-provider.py')
with tempfile.TemporaryDirectory(prefix='semantier-evaluator-unit-') as home:
 p=pathlib.Path(home);env=p/'fixture.env';env.write_text('DASHSCOPE_API_KEY=synthetic-unit-key\nDASHSCOPE_BASE_URL=https://127.0.0.1/v1\n')
 os.environ['DASHSCOPE_API_KEY']='synthetic-unit-key';os.environ['DASHSCOPE_BASE_URL']='https://127.0.0.1/v1';os.environ['SEMANTIER_EVAL_OUTPUT']=home;os.environ['SEMANTIER_EVAL_CREDENTIALS_ENV']=str(env)
 spec=importlib.util.spec_from_file_location('observer',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 class ObserverTests(unittest.TestCase):
  def test_single_wrapper_and_discovery_allowed(self):
   self.assertIsNone(m.forbidden_tool([{'function':{'name':'tool_search','arguments':'{"query":"visible page"}'}}]))
   self.assertIsNone(m.forbidden_tool([{'function':{'name':'tool_call','arguments':json.dumps({'name':'desktop_preview','arguments':{'action':'read'}})}}]))
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

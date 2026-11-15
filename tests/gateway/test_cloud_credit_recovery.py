from types import SimpleNamespace
import pytest
import httpx
import openai
from agent.desktop_browser_target import BrowserTurnState,prelude,recovery_text
from agent.api_error_summary import ApiErrorSummaryMixin


def denied(code,status=402):
    return openai.APIStatusError('Cloud refused request',response=httpx.Response(status,request=httpx.Request('POST','https://cloud.semantier.cn/v1/chat/completions')),body={'detail':{'code':code}})


@pytest.mark.parametrize('code,expected',[('cloud_credit_insufficient','CLOUD_TOP_UP_REQUIRED'),('cloud_billing_reconciliation_required','CLOUD_BILLING_REVIEW_REQUIRED'),('cloud_budget_exceeded','CLOUD_BUDGET_REQUIRED')])
def test_financial_denial_does_not_become_browser_target_failure(monkeypatch,code,expected):
    def reject(*args):raise denied(code)
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent',reject)
    s=BrowserTurnState(session_id='s',generation=1,turn_id='t',target_kind='UNRESOLVED')
    assert list(prelude(s,SimpleNamespace(),'What is your model?',[]))==[]
    assert s.native_failure['code']==expected
    assert 'Billing' in recovery_text(s,'en')
    assert '浏览器' not in recovery_text(s,'zh')
    assert 'browser target' not in ApiErrorSummaryMixin._summarize_api_error(denied(code))


def test_credit_error_explicitly_tells_customer_to_top_up():
    text=ApiErrorSummaryMixin._summarize_api_error(denied('cloud_credit_insufficient'))
    assert 'top up' in text.lower() and 'Billing' in text


def test_reconciliation_does_not_claim_top_up_will_fix_it():
    text=ApiErrorSummaryMixin._summarize_api_error(denied('cloud_billing_reconciliation_required',403))
    assert 'review' in text.lower() and 'top up' not in text.lower()

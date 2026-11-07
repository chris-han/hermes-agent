import pytest
from hermes_state import SessionDB


def test_internal_result_pins_host_metadata_with_physical_identity(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('original', 'api_server')
        args = dict(result_id='effect', result_hash='hash', author_bot_id='bot-owner',
                    result_metadata={'author_display_name':'My customized agent',
                                     'meeting_context':{'case_id':'case','owner_user_id':'owner'}})
        row_id = db.append_internal_result('original','Invitation',**args)
        assert db.append_internal_result('original','Invitation',**args) == row_id
        with pytest.raises(ValueError, match='INTERNAL_RESULT_IDENTITY_CONFLICT'):
            db.append_internal_result('original','Invitation',**{**args,
                'result_metadata':{'author_display_name':'Changed author'}})
        with pytest.raises(ValueError):
            db.append_internal_result('original','Forged',result_id='forged',result_hash='hash',
                result_metadata={'author_bot_id':'forged'})
        rows = db.get_messages('original')
        assert len(rows) == 1 and rows[0]['id'] == row_id
        assert rows[0]['display_metadata']['author_display_name'] == 'My customized agent'
        assert rows[0]['display_metadata']['author_bot_id'] == 'bot-owner'

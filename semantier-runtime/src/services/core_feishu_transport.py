"""Narrow text-DM SDK adapter; no native gateway/session imports or hidden sends."""
import hashlib
import json
import re
import time
from datetime import datetime,timezone
from uuid import uuid5,NAMESPACE_URL


class KnownNotSent(ValueError):
    pass


class CoreFeishuTransport:
    def __init__(self,config,*,endpoint=None):
        from services.meeting_clarification_feishu import FeishuClarificationTransport
        # Existing scoped SDK client owner; no ambient bot or client credentials.
        self.sdk=FeishuClarificationTransport(config,endpoint=endpoint)
        self.config=dict(config)
        self.client=self.sdk.client
        self.client._config.timeout=10

    def receive(self,body,headers,accept):
        from lark_oapi import EventDispatcherHandler
        from lark_oapi.core.model.raw_request import RawRequest
        stamp=headers.get('X-Lark-Request-Timestamp','')
        if len(body)>65536:raise PermissionError('oversized channel callback')
        result=[];errors=[]
        def verified(data):
            if not stamp.isdigit() or abs(time.time()-int(stamp))>300:raise PermissionError('stale channel callback')
            h=data.header;e=data.event;m=e.message;s=e.sender
            if (h.app_id!=self.config['app_id'] or h.tenant_key!=self.config['tenant_key']
                or s.sender_type!='user' or m.chat_type!='p2p' or m.message_type!='text'):
                raise PermissionError('unsupported channel scope or message')
            content=json.loads(m.content)
            if set(content)!={'text'} or not isinstance(content['text'],str):raise PermissionError('text content required')
            value={'event_id':h.event_id,'message_id':m.message_id,'sender_id':s.sender_id.open_id,
                   'chat_id':m.chat_id,'text':content['text'],'created_at':datetime.fromtimestamp(int(m.create_time)/1000,timezone.utc).isoformat(),'source_created_ms':int(m.create_time),
                   'parent_id':m.parent_id or ''}
            if not all(isinstance(value[k],str) and re.fullmatch(r'[A-Za-z0-9_-]{1,256}',value[k]) for k in ('event_id','message_id','sender_id','chat_id')):
                raise PermissionError('bounded transport identifiers required')
            if not all(value[k].startswith(prefix) for k,prefix in (('message_id','om_'),('sender_id','ou_'),('chat_id','oc_'))):raise PermissionError('platform identifiers required')
            result.append(accept(value))
        def guarded(data):
            try:verified(data)
            except Exception as exc:
                errors.append(exc);raise
        handler=EventDispatcherHandler.builder(self.config['encrypt_key'],self.config['verification_token']).register_p2_im_message_receive_v1(guarded).build()
        request=RawRequest();request.uri='/core-feishu';request.body=body;request.headers=dict(headers)
        response=handler.do(request)
        if errors:raise errors[0]
        if response.status_code==200 and not result:
            challenge=json.loads(response.content)
            if set(challenge)=={'challenge'} and isinstance(challenge['challenge'],str) and len(challenge['challenge'])<=200:return challenge
        if response.status_code!=200 or len(result)!=1:
            raise PermissionError('SDK signature/token callback rejected')
        return result[0]

    def reply(self,payload,key):
        from lark_oapi.api.im.v1 import ReplyMessageRequest,ReplyMessageRequestBody
        if (payload.get('purpose')!='channel_chat_reply' or payload.get('account_id')!=self.config.get('core_channel_id')
            or payload.get('bot_generation')!=self.sdk.generation):raise KnownNotSent('scoped channel SDK binding mismatch')
        text=payload['text']
        if not isinstance(text,str) or not text or len(text.encode())>10000:
            raise KnownNotSent('reply exceeds selected text-DM limit')
        request=ReplyMessageRequest.builder().message_id(payload['reply_to']).request_body(
            ReplyMessageRequestBody.builder().msg_type('text').content(json.dumps({'text':text},ensure_ascii=False)).reply_in_thread(False).uuid(str(uuid5(NAMESPACE_URL,key))).build()).build()
        response=self.client.im.v1.message.reply(request)
        data=response.data
        if (not response.success() or data is None or not data.message_id
            or data.chat_id!=payload['chat_id'] or data.parent_id!=payload['reply_to']):
            raise RuntimeError('SDK reply outcome unproven')
        return data.message_id


def probe_bot(config,*,endpoint=None):
    """Validate current credentials through the public SDK, never a native adapter."""
    from lark_oapi import BaseRequest,HttpMethod,AccessTokenType
    transport=CoreFeishuTransport(config,endpoint=endpoint)
    request=BaseRequest.builder().http_method(HttpMethod.GET).uri('/open-apis/bot/v3/info').token_types({AccessTokenType.TENANT}).build()
    response=transport.client.request(request)
    if not response.success():return None
    data=json.loads(response.raw.content).get('bot') or {}
    return {'bot_open_id':data.get('open_id'),'bot_name':data.get('app_name','')}

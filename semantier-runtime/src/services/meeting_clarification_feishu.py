"""Explicit real Feishu SDK boundary. No ambient profile, native runner or client cache."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json

from contracts.meeting_delivery import bot_generation

_SEAL = object()


@dataclass(frozen=True)
class VerifiedIngress:
    case_id: str
    sender: str
    app_id: str
    tenant_key: str
    message_id: str
    chat_id: str
    chat_type: str
    parent_id: str
    text: str
    generation: str
    proof_hash: str
    _seal: object = field(repr=False, compare=False)

    def authentic(self):
        return self._seal is _SEAL


class FeishuClarificationTransport:
    def __init__(self, config, *, endpoint=None):
        from lark_oapi import Client, LogLevel

        self.config = dict(config)
        self.generation = bot_generation(config)
        domain = endpoint or (
            "https://open.larksuite.com"
            if config["domain"] == "lark"
            else "https://open.feishu.cn"
        )
        self.endpoint = domain
        self.client = (
            Client.builder()
            .app_id(config["app_id"])
            .app_secret(config["app_secret"])
            .domain(domain)
            .log_level(LogLevel.ERROR)
            .build()
        )

    def send(self, payload, key):
        from feishu_meeting_coordinator.scripts.feishu_bot_api import (
            send_attendee_message,
        )

        return send_attendee_message(
            attendee_open_ids=[payload["recipient_id"]],
            message=payload["text"],
            idempotency_key=key,
            client=self.client,
        )

    def get_message(self, message_id):
        from feishu_meeting_coordinator.scripts.feishu_bot_api import get_message

        return get_message(message_id=message_id, client=self.client)

    def receive(self, service, case_id, body, headers):
        from lark_oapi import EventDispatcherHandler
        from lark_oapi.core.model.raw_request import RawRequest
        from feishu_meeting_coordinator.gateway import submit_negotiation_reply

        if len(body) > 65536:
            raise PermissionError("SDK callback too large")
        a = service.binding(case_id)
        _, config = service._authority(a, pending=False)
        if bot_generation(config) != self.generation:
            raise PermissionError("stale SDK event configuration")
        timestamp = headers.get("X-Lark-Request-Timestamp", "")
        if (
            not timestamp.isdigit()
            or abs(datetime.now(timezone.utc).timestamp() - int(timestamp)) > 300
        ):
            raise PermissionError("stale SDK callback")
        outcome = []

        def accept(data):
            event = data.event
            message = event.message
            sender = event.sender.sender_id
            if message.message_type != "text":
                raise PermissionError("structured text only")
            header = data.header
            proof = VerifiedIngress(
                case_id=case_id,
                sender=sender.open_id,
                app_id=header.app_id,
                tenant_key=header.tenant_key,
                message_id=message.message_id,
                chat_id=message.chat_id,
                chat_type=message.chat_type,
                parent_id=message.parent_id or "",
                text=json.loads(message.content)["text"],
                generation=self.generation,
                proof_hash=hashlib.sha256(body).hexdigest(),
                _seal=_SEAL,
            )
            outcome.append(
                submit_negotiation_reply(
                    {"negotiation_id": case_id}, store=service.store, core_ingress=proof
                )
            )

        handler = (
            EventDispatcherHandler.builder(
                config["encrypt_key"], config["verification_token"]
            )
            .register_p2_im_message_receive_v1(accept)
            .build()
        )
        request = RawRequest()
        request.uri = "/meeting-clarification"
        request.body = body
        request.headers = dict(headers)
        result = handler.do(request)
        if result.status_code != 200 or len(outcome) != 1:
            raise PermissionError("SDK callback rejected or ambiguous")
        return outcome[0]

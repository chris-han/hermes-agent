"""Opt-in, bounded tool-call-only stream evidence for one admitted attempt.

This is diagnostic evidence, not a parser used to make execution decisions.
"""
from __future__ import annotations

import json
import os
import re
import stat

_LIMIT = 8 * 1024 * 1024


class ToolStreamTrace:
    def __init__(self, directory, attempt_id, boundary, *, content_encoding=''):
        self.enabled = False
        self.events = []
        self.buffer = b''
        self.size = 0
        self.truncated = False
        self.encoding = content_encoding
        self.directory, self.attempt_id, self.boundary = directory, attempt_id, boundary
        if not directory or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', attempt_id or ''):
            return
        if boundary not in {'provider', 'forwarded'}:
            return
        try:
            info = os.lstat(directory)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                return
            self.enabled = True
        except OSError:
            return

    def feed(self, data):
        if not self.enabled or self.truncated or self.encoding not in ('', 'identity'):
            return
        self.size += len(data)
        if self.size > _LIMIT:
            self.truncated = True
            self.buffer = b''
            return
        self.buffer += data
        # SSE records are separated by a blank line, including CRLF streams.
        self.buffer = self.buffer.replace(b'\r\n', b'\n')
        while b'\n\n' in self.buffer:
            event, self.buffer = self.buffer.split(b'\n\n', 1)
            self._event(event)

    def _event(self, event):
        lines = [line[5:].lstrip(b' ') for line in event.split(b'\n') if line.startswith(b'data:')]
        if not lines:
            return
        raw = b'\n'.join(lines)
        if raw == b'[DONE]':
            self.events.append({'done': True})
            return
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError()
        except (ValueError, UnicodeDecodeError, RecursionError):
            self.events.append({'invalid_event': True})
            return
        selected = {key: obj[key] for key in ('id', 'model', 'object') if key in obj}
        event_type = obj.get('type')
        if isinstance(event_type, str):
            selected['type'] = event_type
            for key in ('output_index', 'item_id', 'sequence_number'):
                if key in obj:
                    selected[key] = obj[key]
            if event_type.startswith('response.function_call_arguments.'):
                for key in ('delta', 'arguments'):
                    if key in obj:
                        selected[key] = obj[key]
            item = obj.get('item')
            if isinstance(item, dict) and item.get('type') == 'function_call':
                selected['item'] = {k: item[k] for k in ('id', 'type', 'call_id', 'name', 'arguments', 'status') if k in item}
            response = obj.get('response')
            if isinstance(response, dict):
                selected['response'] = {k: response[k] for k in ('id', 'status', 'model') if k in response}
                details = response.get('incomplete_details')
                if isinstance(details, dict):
                    selected['response']['incomplete_details'] = {k: details[k] for k in ('reason',) if k in details}
                elif 'incomplete_details' in response and details is None:
                    selected['response']['incomplete_details'] = None
        if 'choices' in obj and isinstance(obj['choices'], list):
            selected['choices'] = []
            for choice in obj['choices']:
                if not isinstance(choice, dict):
                    continue
                out = {k: choice[k] for k in ('index', 'finish_reason') if k in choice}
                delta = choice.get('delta')
                if isinstance(delta, dict) and isinstance(delta.get('tool_calls'), list):
                    out['tool_calls'] = []
                    for call in delta['tool_calls']:
                        if not isinstance(call, dict):
                            continue
                        item = {k: call[k] for k in ('index', 'id', 'type') if k in call}
                        if isinstance(call.get('function'), dict):
                            item['function'] = {k: call['function'][k] for k in ('name', 'arguments') if k in call['function']}
                        out['tool_calls'].append(item)
                selected['choices'].append(out)
        self.events.append(selected)

    def finish(self, termination, *, status=None, request_id=None, error_type=None):
        if not self.enabled:
            return
        self.enabled = False
        result = {'attempt_id': self.attempt_id, 'boundary': self.boundary,
                  'termination': termination, 'status': status, 'request_id': request_id, 'error_type': error_type,
                  'content_encoding': self.encoding, 'limit_exceeded': self.truncated,
                  'incomplete_event': bool(self.buffer.strip()), 'events': self.events}
        try:
            # O_NOFOLLOW and directory ownership/mode prevent replacing a target
            # with a symlink or writing sensitive fragments to a shared directory.
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(directory_fd)
                if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    return
                name = self.attempt_id + '.' + self.boundary + '.json'
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
                with os.fdopen(fd, 'w', encoding='utf-8') as output:
                    json.dump(result, output, ensure_ascii=False, separators=(',', ':'))
            finally:
                os.close(directory_fd)
        except OSError:
            pass  # Diagnostics must never affect the request or leak to logs.


def trace_for(attempt_id, boundary, **kwargs):
    selected = os.environ.get('SEMANTIER_TOOL_TRACE_ATTEMPT')
    directory = os.environ.get('SEMANTIER_TOOL_TRACE_DIR')
    if selected == 'next':
        candidate = ToolStreamTrace(directory, attempt_id, boundary, **kwargs)
        if not candidate.enabled:
            return candidate
        try:
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(directory_fd)
                if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    candidate.enabled = False
                    return candidate
                try:
                    fd = os.open('.tool-trace-claim', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
                    with os.fdopen(fd, 'w') as claim:
                        claim.write(attempt_id)
                except FileExistsError:
                    fd = os.open('.tool-trace-claim', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                    with os.fdopen(fd) as claim:
                        if claim.read(201) != attempt_id:
                            candidate.enabled = False
            finally:
                os.close(directory_fd)
        except OSError:
            candidate.enabled = False
        return candidate
    if not selected or selected != attempt_id:
        return ToolStreamTrace(None, '', boundary)
    return ToolStreamTrace(directory, attempt_id, boundary, **kwargs)

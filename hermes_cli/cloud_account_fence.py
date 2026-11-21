"""Private native helper for the Desktop account restriction; no credentials cross stdin."""
import json
import sys

from hermes_cli.cloud_device_client import set_account_fence


def main():
    try:
        wire = sys.stdin.read(8193)
        if len(wire.encode('utf-8')) > 8192:
            raise ValueError('request too large')
        value = json.loads(wire)
        if not isinstance(value, dict) or set(value) != {'account'}:
            raise ValueError('invalid request')
        set_account_fence(value['account'])
        print('{"ok":true}')
        return 0
    except Exception:
        print('{"ok":false}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

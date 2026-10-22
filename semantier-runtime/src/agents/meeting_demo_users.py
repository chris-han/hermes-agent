"""Explicit local-demo identity seeding. Deployment startup must explicitly enable workshop demo provisioning."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib.parse import quote

from agents import auth_db
from agents.gateway_identity import hash_local_password, verify_local_password_hash

DEMO_PASSWORD = 'SemantierDemo!2026'
DEMO_SCOPE = 'meeting-coordinator-multi-workspace'
DEMO_SEED_VERSION = 'meeting-demo-users.v1'
DEMO_ROSTER = (
    ('alex-chen', 'Alex Chen'), ('maya-patel', 'Maya Patel'), ('daniel-kim', 'Daniel Kim'),
    ('sofia-martinez', 'Sofia Martinez'), ('liam-obrien', "Liam O'Brien"), ('aisha-rahman', 'Aisha Rahman'),
    ('noah-williams', 'Noah Williams'), ('emma-johnson', 'Emma Johnson'), ('lucas-silva', 'Lucas Silva'),
    ('hana-sato', 'Hana Sato'), ('omar-haddad', 'Omar Haddad'), ('grace-li', 'Amy Q'),
)


def demo_avatar(index: int, name: str) -> str:
    initials = ''.join(word[0] for word in name.split())[:2]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
           f'<rect width="64" height="64" rx="32" fill="hsl({index * 29},45%,32%)"/>'
           f'<text x="32" y="40" text-anchor="middle" font-family="sans-serif" font-size="23" fill="white">{initials}</text></svg>')
    return 'data:image/svg+xml,' + quote(svg, safe='')


def seed_meeting_demo_users(*, local_demo: bool = False, preserve_existing: bool = False, password: str = DEMO_PASSWORD) -> list[dict]:
    if local_demo is not True:
        raise ValueError('EXPLICIT_LOCAL_DEMO_REQUIRED')
    auth_db.ensure_auth_db()
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    if auth_db.is_postgresql():
        with auth_db.identity_write_transaction():
            current=auth_db.load_users()
            result=_seed_roster(current,now,preserve_existing,password,lambda key,user: current.__setitem__(key,user))
            auth_db.save_users(current)
            return result
    with auth_db._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = {row['user_id']: json.loads(row['payload_json']) for row in conn.execute('SELECT * FROM users')}
        def persist(key,user):
            conn.execute('INSERT INTO users(user_id,payload_json,updated_at) VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET payload_json=excluded.payload_json,updated_at=excluded.updated_at',
                         (key,json.dumps(user,sort_keys=True),user['updated_at']))
        return _seed_roster(current,now,preserve_existing,password,persist)


def _seed_roster(current,now,preserve_existing,password,persist):
    # Check the complete roster before writing any record.
    for slug, _ in DEMO_ROSTER:
        user_id = 'demo-user-' + slug
        matching = [u for u in current.values() if u.get('password_login_name') == 'demo-' + slug or u.get('workspace_id') == 'demo-ws-' + slug]
        for candidate in matching + ([current[user_id]] if user_id in current else []):
            if candidate.get('user_id') != user_id or candidate.get('demo_scope') != DEMO_SCOPE or not candidate.get('demo_account'):
                raise ValueError('DEMO_IDENTITY_COLLISION:' + slug)
    result = []
    for index, (slug, name) in enumerate(DEMO_ROSTER):
        user_id = 'demo-user-' + slug; workspace_id = 'demo-ws-' + slug
        prior = current.get(user_id, {})
        if preserve_existing and prior:
            result.append(prior)
            continue
        password_hash = prior.get('password_hash', '')
        if not verify_local_password_hash(password_hash, password):
            password_hash = hash_local_password(password)
        user = {**prior, 'user_id': user_id, 'workspace_id': workspace_id, 'workspace_owner_id': user_id,
                'workspace_slug': workspace_id, 'password_login_name': 'demo-' + slug,
                'password_hash': password_hash, 'auth_provider': 'password', 'name': name, 'display_name': name,
                'avatar_url': demo_avatar(index, name), 'feishu_open_id': prior.get('feishu_open_id') or 'ou_demo_' + slug.replace('-', '_'),
                'demo_account': True, 'demo_scope': DEMO_SCOPE, 'demo_seed_version': DEMO_SEED_VERSION,
                'profile_completed_at': prior.get('profile_completed_at') or now,
                'created_at': prior.get('created_at') or now, 'updated_at': prior.get('updated_at') or now}
        if user != prior:
            persist(user_id,user)
        result.append(user)
    return result

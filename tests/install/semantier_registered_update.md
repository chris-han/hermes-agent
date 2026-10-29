# Registered Semantier acceptance transaction

Entry point: `semantier_registered_update.py`, native Windows Python 3.11+ (standard library only). It uses the existing MSIX reader, native package/process lifecycle and EN/ZH UIA helper. The old isolated guard/preflight and JS journal remain diagnostic historical work; they do not authorize this transaction.

Modes: `preflight`, `baseline`, `upgrade`, `reconcile`, `cleanup`, each with `--request C:\absolute\request.json`. Preflight reads OS/package/artifact state and creates no journal, package or data directories. Actual mutation remains subject to the operator's separate authorization and an authenticated disposable Windows user. The request SID is a constraint on that external authorization. Normal ChrisH name and the observed normal-user SID are refused.

The operator must supply a real disposable active interactive x64 session, an existing user-owned directory beneath `%USERPROFILE%\semantier-acceptance\RUN`, restrictive ACL (only that SID, SYSTEM and Administrators may mutate it), readable signed packages and the actual external qualified target facts. No certificate import or trust modification is performed. Existing acceptance registration is refused for a new transaction.

Request shape (replace placeholders from actual observations; never put credential values in it):

```json
{
  "authorized_sid": "OPERATOR_AUTHORIZED_DISPOSABLE_SID",
  "root": "C:\\Users\\DISPOSABLE\\semantier-acceptance\\RUN",
  "expected_language": "zh",
  "expected_profiles": ["semantier-local"],
  "baseline": {
    "path": "D:\\repo\\deliverables\\SemantierAgent-profile-channels-20261028T105418442160Z\\signed\\distribution\\SemantierAgent-0.0.58-win-x64.msix",
    "sha256": "55be24a18b1827cabfd951e30645a69e9cfcd0d1b07729e1b1c8f3483b079ec9"
  },
  "target": {
    "path": "QUALIFIED_HIGHER_VERSION_SIGNED_MSIX",
    "sha256": "ACTUAL_QUALIFIED_SHA256",
    "manifest_sha256": "ACTUAL_SELECTED_ORIGIN_MANIFEST_SHA256",
    "chunk_descriptor_sha256": "ACTUAL_QUALIFIED_DESCRIPTOR_SHA256"
  },
  "sentinels": {
    "chats": ["RELATIVE_REAL_CHAT_SENTINEL"],
    "credentials": ["RELATIVE_REAL_CREDENTIAL_SENTINEL"],
    "providers": ["RELATIVE_REAL_PROVIDER_SENTINEL"],
    "memories": ["RELATIVE_REAL_MEMORY_SENTINEL"],
    "skills": ["RELATIVE_REAL_SKILL_SENTINEL"],
    "language": ["RELATIVE_PRODUCTION_PREFERENCE_SENTINEL"],
    "theme": ["RELATIVE_PRODUCTION_THEME_SENTINEL"],
    "profiles": ["userdata/active-profile.json"],
    "legacy_backup": ["RELATIVE_RECOVERABLE_LEGACY_BACKUP_SENTINEL"]
  }
}
```

Use Windows Python with `-I` and the exact staged source entrypoint. Example read-only command (the current normal user must refuse):

```powershell
& 'C:\Users\ChrisH\AppData\Local\Programs\Python\Python312\python.exe' -I 'D:\repo\qualification-private\W1-CANDIDATE\semantier_registered_update.py' preflight --request 'C:\Users\DISPOSABLE\semantier-acceptance\RUN\request.json'
```

Do not execute `baseline`, `upgrade` or `cleanup` until authorization and prerequisites exist. After supported preflight, authorized sequencing is `baseline` (retains registration and launches standalone local backend), observe/seed the real data and preference fixtures, qualified console publication, normal source selection/check controls, then `upgrade`. The baseline readiness observation is durably recorded before upgrade is allowed. UIA clicks production controls; there is no apply IPC or fake Actions environment. The target must be strictly newer and the actual registered family must match.

`reconcile` reads pending intent and OS state. Exact registration success with interrupted recording is observed without repeating it. Successful package installation with no relaunch is recorded partial and never passed. Ambiguous installer/process states hold for inspection. `cleanup` removes only a registration proven by recorded register/update outcomes and exact current identities. It normally closes the recorded main process and stops only a lifecycle-bound recorded gateway using held OS process handles. Unknown processes, changed identities, torn records, child links or unexpected registrations hold. Data, logs, journal and backups are retained; directory deletion is intentionally not part of package cleanup.

Every consequential action has a flushed JSONL intent before native invocation, native precondition readback, and an observed outcome or uncertainty. Root identity is held across the transaction; child journal/lock links are refused. OS locks release on process death; locks are never stolen by age. Do not edit the journal to force continuation. Interrupted UIA may have opened App Installer; unresolved native state is preserved, including when no safe retry can be proved.

Evidence limits: the handoff watcher proves the full downloaded MSIX hash/signature and the production manifest-cache binding if captured before cache retirement. It does not observe individual chunk transfers or independently hash the descriptor; the descriptor value is explicitly an expected qualification input. Missed custody readback fails. Readiness observes local backend health, actual UIA language control, saved active profile and physical namespace. It does not prove the actual discovered roster, ready composer placeholder or rendered theme. Required category sentinels prove selected bytes, not their semantic completeness. Those original acceptance gates remain separately unverified until the real session and original matrices establish them. A failed automatic relaunch can be reconciled/cleaned but cannot be converted into success by launching the successor manually.

No positive native registration, update, removal or acceptance has been executed to qualify this candidate. Windows P05 remains FAILED.


The native owner additionally resolves the active WTS session's domain/user to its OS SID and requires it to equal the process/request SID. A secondary logon on another user's desktop does not establish this requirement. The journal binds `session_sid` alongside process SID, session ID and physical root. Historical transactions lacking this new binding require inspection and remain held; they are never silently adopted as current-context evidence. This is separate from producer/layout/native qualification and does not pass any native acceptance gate.

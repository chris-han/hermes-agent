"""Finite workspace-owned database paths eligible for native scope cohabitation.

A suffix only locates a candidate home. The native registry or frozen deployment
map must independently establish its governed identity before a schema is used.
"""
from pathlib import Path

# Exact paths defined by these storage owners; no glob, filename-only match,
# legacy EOS events alias, or arbitrary ancestor walk is accepted.
AUXILIARY_SUFFIXES=(
    Path('.semantier/runtime-detection/snapshots.sqlite3'),
    Path('.semantier/semantic-feedback/feedback.sqlite3'),
    Path('.semantier/canonical-list/profiles.sqlite3'),
    Path('.semantier/mvl/selection.sqlite3'),
    Path('.semantier/graph-schema-learning/learning.sqlite3'),
    Path('.semantier/graph-schema-discovery/discovery.sqlite3'),
    Path('runs/ontology-building/ontology-builds.sqlite'),
    Path('artifacts/graph_intent_publication_manifest.sqlite3'),
)


def resolve_auxiliary_scope(path):
    """Return a validated native SessionScope; a path never supplies a tenant."""
    from hermes_state_backend import resolve_postgres_scope
    target=Path(path).expanduser().resolve()
    for suffix in AUXILIARY_SUFFIXES:
        if target.parts[-len(suffix.parts):]==suffix.parts:
            home=target.parents[len(suffix.parts)-1]
            return resolve_postgres_scope(home)
    raise ValueError('database is not an approved workspace auxiliary path')


def auxiliary_schema_sql():
    """Frozen migration asset, also installed by the guarded signup allocator."""
    return '\n'.join(path.read_text() for path in auxiliary_schema_assets())


def auxiliary_schema_assets():
    """Finite checked-in owner assets, in dependency order."""
    return tuple(Path(__file__).with_name(name) for name in (
        'hosted_standalone.sql','ontology_build_postgres.sql',
        'tender_journey_postgres.sql','hosted_eos_workflow.sql',
        'graph_intent_brief_postgres.sql',
        'contextgraph_manifest_postgres.sql',
        'tender_gold_postgres.sql',
    ))

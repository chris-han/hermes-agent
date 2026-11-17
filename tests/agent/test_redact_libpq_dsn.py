"""Operator/log diagnostics must not expose libpq keyword credentials."""
import logging

import pytest

from agent.redact import RedactingFormatter, redact_sensitive_text, redact_terminal_output


@pytest.mark.parametrize('password', ['synthetic-credential-123', "'synthetic credential \\' quoted'", r'synthetic\ credential'])
def test_keyword_dsn_password_is_scrubbed_on_all_diagnostic_surfaces(password):
    text = f'connection user=app password={password} dbname=dev host=127.0.0.1 port=5432'
    record = logging.LogRecord('diagnostic', logging.ERROR, __file__, 1, text, (), None)
    outputs = [redact_sensitive_text(text, force=True, code_file=True),
               redact_terminal_output(text, 'python diagnostic.py'), RedactingFormatter().format(record)]
    for output in outputs:
        assert 'synthetic' not in output
        assert 'dbname=dev' in output and 'host=127.0.0.1' in output


def test_ordinary_password_prose_and_url_query_policy_are_unchanged():
    for text in ['explain password=example in the form', 'https://example.test/?user=app&password=example']:
        assert redact_sensitive_text(text, force=True, code_file=True) == text

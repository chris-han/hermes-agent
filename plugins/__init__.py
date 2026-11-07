"""Hermes plugins, composable with host-owned plugins on the import path."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

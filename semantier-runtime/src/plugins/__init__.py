"""Semantier-managed Hermes plugin sources tracked in git.

Semantier and Hermes share the ``plugins`` import namespace. Extend this
package so Hermes bundled providers such as ``plugins.browser`` remain
importable when both repository roots are on ``sys.path``.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

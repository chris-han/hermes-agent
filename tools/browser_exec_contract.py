"""Pure preflight of the installed browser-harness interface, not a Python sandbox."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
from pathlib import Path
import symtable
import sys
from typing import Any


def _signature(arguments):
    safe = copy.deepcopy(arguments)
    for arg in [*safe.posonlyargs, *safe.args, *safe.kwonlyargs, *([safe.vararg] if safe.vararg else []), *([safe.kwarg] if safe.kwarg else [])]:
        arg.annotation = None
    safe.defaults = [ast.Constant(value=Ellipsis) for _ in safe.defaults]
    safe.kw_defaults = [ast.Constant(value=Ellipsis) if value is not None else None for value in safe.kw_defaults]
    return ast.unparse(safe)


def _source_namespace(path: Path) -> tuple[dict, set, dict, str]:
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    symbols = symtable.symtable(source, str(path), 'exec')
    names = {s.get_name() for s in symbols.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    helpers, methods = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
            helpers[node.name] = {'signature': _signature(node.args)}
            methods[node.name] = sorted({call.args[0].value for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'cdp'
                and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)})
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != '*')
    return helpers, names, methods, source


def installed_api(workspace: str | Path | None = None) -> dict[str, Any]:
    """Read definitions without importing helpers (.env/custom helper imports execute code)."""
    spec = importlib.util.find_spec('browser_harness')
    if spec is None or not spec.origin:
        raise RuntimeError('Installed browser-harness source is unavailable')
    root = Path(spec.origin).parent
    helpers, names, methods, source = _source_namespace(root / 'helpers.py')
    _, runtime_names, _, runtime_source = _source_namespace(root / 'run.py')
    # These are exports of the actual import-star source, including recorder imports.
    for imported in ast.parse(source).body:
        if isinstance(imported, ast.ImportFrom) and imported.level and imported.module == 'recorder':
            recorders, _, _, recorder_source = _source_namespace(root / 'recorder.py')
            source += recorder_source
            for alias in imported.names:
                if alias.name in recorders:
                    helpers[alias.asname or alias.name] = recorders[alias.name]
    if workspace is not None:
        custom = Path(workspace) / 'agent_helpers.py'
        if custom.is_file():
            extra, extra_names, extra_methods, extra_source = _source_namespace(custom)
            helpers.update(extra); names.update(extra_names); methods.update(extra_methods)
            source += extra_source
    return {'module': 'browser_harness.helpers', 'version': importlib.metadata.version('browser-harness'),
            'source_sha256': hashlib.sha256((source + runtime_source).encode()).hexdigest(),
            'helpers': helpers, 'exports': sorted(names), 'globals': sorted(names | runtime_names | set(vars(builtins))), 'cdp_methods': methods}


def _module_spec(name: str):
    """Resolve builtin/frozen/package paths without importing a parent package."""
    loaded = getattr(sys.modules.get(name), '__spec__', None)
    if loaded is not None and loaded.origin in ('built-in', 'frozen'):
        return loaded
    search = None
    spec = None
    parts = name.split('.')
    for index in range(len(parts)):
        full = '.'.join(parts[:index + 1])
        spec = (importlib.machinery.BuiltinImporter.find_spec(full)
                or importlib.machinery.FrozenImporter.find_spec(full)
                or importlib.machinery.PathFinder.find_spec(full, search))
        if spec is None:
            return None
        search = spec.submodule_search_locations
        if index < len(parts) - 1 and search is None:
            return None
    return spec


def _error(code: str, *, api: dict, **details) -> dict:
    return {'success': False, 'code': code, 'error': code, 'path': 'arguments.code',
            **details, 'helper_module': api.get('module'), 'supported_helpers': sorted(api.get('helpers', {})),
            'api_source_sha256': api.get('source_sha256'), 'recovery': {'same_context': True,
            'next_action': 'correct_code_using_installed_interface', 'automatic_install': False}, 'retryable': True}


def _import_error(node, api):
    modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or '']
    if isinstance(node, ast.ImportFrom) and node.level:
        modules = [importlib.util.resolve_name('.' * node.level + (node.module or ''), 'browser_harness')]
    for module in modules:
        if _module_spec(module) is None:
            requested = [alias.name for alias in node.names] if isinstance(node, ast.ImportFrom) else []
            code = 'UNSUPPORTED_HELPER_IMPORT' if any(name in api['helpers'] for name in requested) else 'IMPORT_UNAVAILABLE'
            return _error(code, api=api, module=module, line=node.lineno, requested_names=requested)
    if isinstance(node, ast.ImportFrom) and modules[0] == api['module'].rsplit('.', 1)[0]:
        spec = _module_spec(modules[0])
        _, exported, _, _ = _source_namespace(Path(spec.origin))
        for alias in node.names:
            if alias.name in api['helpers'] and alias.name not in exported and _module_spec(modules[0] + '.' + alias.name) is None:
                return _error('UNSUPPORTED_HELPER_IMPORT', api=api, module=modules[0], helper=alias.name, line=node.lineno)
    if isinstance(node, ast.ImportFrom) and modules[0] == api['module']:
        for alias in node.names:
            if alias.name != '*' and alias.name not in api['exports']:
                return _error('UNKNOWN_HELPER', api=api, helper=alias.name, line=node.lineno)
    return None


def _global_references(table):
    found = {s.get_name() for s in table.get_symbols() if s.is_referenced() and s.is_global()}
    for child in table.get_children():
        found.update(_global_references(child))
    return found


def _aliases(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname or a.name.split('.')[0]: a.name if a.asname else a.name.split('.')[0] for a in node.names})
        if isinstance(node, ast.ImportFrom):
            aliases.update({a.asname or a.name: (node.module or '') + '.' + a.name for a in node.names if a.name != '*'})
    return aliases


def _qualified(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _qualified(node.value, aliases) + '.' + node.attr
    return ''


def _lexical_names(node, parents, tables, module):
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child in parent.body:
            table = tables.get((parent.name, parent.lineno), module)
            break
        if isinstance(parent, ast.Lambda) and child is parent.body:
            table = tables.get(('lambda', parent.lineno), module)
            break
        child = parent
    else:
        table = module
    return {symbol.get_name() for symbol in table.get_symbols()
            if (symbol.is_local() or symbol.is_free() or symbol.is_nonlocal()) and not symbol.is_imported()}


def _scope_tables(table):
    found = {}
    for child in table.get_children():
        found[(child.get_name(), child.get_lineno())] = child
        found.update(_scope_tables(child))
    return found


def _lookup_error(node, qualified, api, aliases):
    if qualified not in ('getattr', 'builtins.getattr') or len(node.args) < 2:
        return None
    if _qualified(node.args[0], aliases) != api['module']:
        return None
    attribute = node.args[1]
    if not isinstance(attribute, ast.Constant) or not isinstance(attribute.value, str):
        return _error('UNVERIFIABLE_HELPER_REFERENCE', api=api, helper='getattr', line=node.lineno)
    if attribute.value not in api['exports']:
        return _error('UNKNOWN_HELPER', api=api, helper=attribute.value, line=node.lineno)
    return None


def _call_error(node, *, api, available, unknown, aliases, unavailable_cdp, lexical=()):
    name = node.func.id if isinstance(node.func, ast.Name) else None
    if name in lexical:
        return None
    aliases = {key: value for key, value in aliases.items() if key not in lexical}
    qualified = _qualified(node.func, aliases)
    lookup_error = _lookup_error(node, qualified, api, aliases)
    if lookup_error is not None:
        return lookup_error
    if name in {'globals', 'locals', 'eval', 'exec', '__import__', 'vars'} and name not in available:
        return _error('UNVERIFIABLE_HELPER_REFERENCE', api=api, helper=name, line=node.lineno)
    if qualified in {'importlib.import_module', 'builtins.eval', 'builtins.exec', 'builtins.__import__'}:
        return _error('UNVERIFIABLE_HELPER_REFERENCE', api=api, helper=qualified, line=node.lineno)
    if name in unknown:
        return _error('UNKNOWN_HELPER', api=api, helper=name, line=node.lineno)
    prefix = api['module'] + '.'
    if qualified.startswith(prefix) and qualified[len(prefix):].split('.')[0] not in api['exports']:
        return _error('UNKNOWN_HELPER', api=api, helper=qualified, line=node.lineno)
    helper = qualified[len(prefix):] if qualified.startswith(prefix) else name
    methods = set(api['cdp_methods'].get(helper, []))
    if helper == 'cdp' and node.args and isinstance(node.args[0], ast.Constant):
        methods.add(node.args[0].value)
    if methods & unavailable_cdp:
        return _error('BACKEND_CAPABILITY_UNAVAILABLE', api=api, helper=helper,
                      cdp_methods=sorted(methods & unavailable_cdp), line=node.lineno)
    return None


def validate_code(code: str, *, api: dict | None = None, workspace=None, unavailable_cdp=()) -> dict | None:
    """Validate explicit imports and statically resolvable helper calls before any dispatch."""
    try:
        api = api if api is not None else installed_api(workspace)
    except (OSError, ImportError, RuntimeError, SyntaxError, ValueError) as error:
        return _error('API_CONTRACT_UNAVAILABLE', api=api or {}, error_type=type(error).__name__)
    try:
        tree = ast.parse(code)
        symbols = symtable.symtable(code, '<browser_exec>', 'exec')
    except (OSError, RuntimeError, SyntaxError, ValueError) as error:
        return _error('INVALID_ARGUMENT' if isinstance(error, (SyntaxError, ValueError)) else 'API_CONTRACT_UNAVAILABLE',
                      api=api or {}, error_type=type(error).__name__, line=getattr(error, 'lineno', None))
    bound = {s.get_name() for s in symbols.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    unknown = _global_references(symbols) - set(api['globals']) - bound
    aliases = _aliases(tree)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    tables = _scope_tables(symbols)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                error = _import_error(node, api)
            except (OSError, ImportError, SyntaxError, ValueError) as error:
                return _error('API_CONTRACT_UNAVAILABLE', api=api, error_type=type(error).__name__, line=node.lineno)
            if error is not None:
                return error
            if isinstance(node, ast.ImportFrom) and any(a.name == '*' for a in node.names) and node.module != api['module']:
                return _error('UNVERIFIABLE_HELPER_REFERENCE', api=api, module=node.module, line=node.lineno)
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            lexical = _lexical_names(node, parents, tables, symbols)
            qualified = _qualified(node, {key: value for key, value in aliases.items() if key not in lexical})
            prefix = api['module'] + '.'
            if isinstance(node, ast.Name) and node.id in unknown and node.id not in lexical:
                return _error('UNKNOWN_HELPER', api=api, helper=node.id, line=node.lineno)
            if qualified.startswith(prefix) and qualified[len(prefix):].split('.')[0] not in api['exports']:
                return _error('UNKNOWN_HELPER', api=api, helper=qualified, line=node.lineno)
        if isinstance(node, ast.Call):
            error = _call_error(node, api=api, available=bound, unknown=unknown, aliases=aliases,
                                unavailable_cdp=set(unavailable_cdp), lexical=_lexical_names(node, parents, tables, symbols))
            if error is not None:
                return error
    return None

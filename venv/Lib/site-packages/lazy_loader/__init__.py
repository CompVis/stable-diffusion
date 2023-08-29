"""
lazy_loader
===========

Makes it easy to load subpackages and functions on demand.
"""
import ast
import importlib
import importlib.util
import inspect
import os
import sys
import types
import warnings

__all__ = ["attach", "load", "attach_stub"]


def attach(package_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows::

      import mysubmodule
      import anothersubmodule

      from .foo import someattr

    The idea is to replace a package's `__getattr__`, `__dir__`, and
    `__all__`, such that all imports work exactly the way they would
    with normal imports, except that the import occurs upon first use.

    The typical way to call this function, replacing the above imports, is::

      __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        ['mysubmodule', 'anothersubmodule'],
        {'foo': ['someattr']}
      )

    This functionality requires Python 3.7 or higher.

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    submodules : set
        List of submodules to attach.
    submod_attrs : dict
        Dictionary of submodule -> list of attributes / functions.
        These attributes are imported as they are used.

    Returns
    -------
    __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    __all__ = sorted(submodules | attr_to_modules.keys())

    def __getattr__(name):
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        elif name in attr_to_modules:
            submod_path = f"{package_name}.{attr_to_modules[name]}"
            submod = importlib.import_module(submod_path)
            attr = getattr(submod, name)

            # If the attribute lives in a file (module) with the same
            # name as the attribute, ensure that the attribute and *not*
            # the module is accessible on the package.
            if name == attr_to_modules[name]:
                pkg = sys.modules[package_name]
                pkg.__dict__[name] = attr

            return attr
        else:
            raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__():
        return __all__

    if os.environ.get("EAGER_IMPORT", ""):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


class DelayedImportErrorModule(types.ModuleType):
    def __init__(self, frame_data, *args, **kwargs):
        self.__frame_data = frame_data
        super().__init__(*args, **kwargs)

    def __getattr__(self, x):
        if x in ("__class__", "__file__", "__frame_data"):
            super().__getattr__(x)
        else:
            fd = self.__frame_data
            raise ModuleNotFoundError(
                f"No module named '{fd['spec']}'\n\n"
                "This error is lazily reported, having originally occured in\n"
                f'  File {fd["filename"]}, line {fd["lineno"]}, in {fd["function"]}\n\n'
                f'----> {"".join(fd["code_context"]).strip()}'
            )


def load(fullname, error_on_import=False):
    """Return a lazily imported proxy for a module.

    We often see the following pattern::

      def myfunc():
          import numpy as np
          np.norm(...)
          ....

    Putting the import inside the function prevents, in this case,
    `numpy`, from being imported at function definition time.
    That saves time if `myfunc` ends up not being called.

    This `load` function returns a proxy module that, upon access, imports
    the actual module.  So the idiom equivalent to the above example is::

      np = lazy.load("numpy")

      def myfunc():
          np.norm(...)
          ....

    The initial import time is fast because the actual import is delayed
    until the first attribute is requested. The overall import time may
    decrease as well for users that don't make use of large portions
    of your library.

    Warning
    -------
    While lazily loading *sub*packages technically works, it causes the
    package (that contains the subpackage) to be eagerly loaded even
    if the package is already lazily loaded.
    So, you probably shouldn't use subpackages with this `load` feature.
    Instead you should encourage the package maintainers to use the
    `lazy_loader.attach` to make their subpackages load lazily.

    Parameters
    ----------
    fullname : str
        The full name of the module or submodule to import.  For example::

          sp = lazy.load('scipy')  # import scipy as sp

    error_on_import : bool
        Whether to postpone raising import errors until the module is accessed.
        If set to `True`, import errors are raised as soon as `load` is called.

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module.  Can be used like any regularly imported module.
        Actual loading of the module occurs upon first attribute request.

    """
    try:
        return sys.modules[fullname]
    except KeyError:
        pass

    if "." in fullname:
        msg = (
            "subpackages can technically be lazily loaded, but it causes the "
            "package to be eagerly loaded even if it is already lazily loaded."
            "So, you probably shouldn't use subpackages with this lazy feature."
        )
        warnings.warn(msg, RuntimeWarning)

    spec = importlib.util.find_spec(fullname)
    if spec is None:
        if error_on_import:
            raise ModuleNotFoundError(f"No module named '{fullname}'")
        else:
            try:
                parent = inspect.stack()[1]
                frame_data = {
                    "spec": fullname,
                    "filename": parent.filename,
                    "lineno": parent.lineno,
                    "function": parent.function,
                    "code_context": parent.code_context,
                }
                return DelayedImportErrorModule(frame_data, "DelayedImportErrorModule")
            finally:
                del parent

    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module

    loader = importlib.util.LazyLoader(spec.loader)
    loader.exec_module(module)

    return module


class _StubVisitor(ast.NodeVisitor):
    """AST visitor to parse a stub file for submodules and submod_attrs."""

    def __init__(self):
        self._submodules = set()
        self._submod_attrs = {}

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level != 1:
            raise ValueError(
                "Only within-module imports are supported (`from .* import`)"
            )
        if node.module:
            attrs: list = self._submod_attrs.setdefault(node.module, [])
            attrs.extend(alias.name for alias in node.names)
        else:
            self._submodules.update(alias.name for alias in node.names)


def attach_stub(package_name: str, filename: str):
    """Attach lazily loaded submodules, functions from a type stub.

    This is a variant on ``attach`` that will parse a `.pyi` stub file to
    infer ``submodules`` and ``submod_attrs``. This allows static type checkers
    to find imports, while still providing lazy loading at runtime.

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    filename : str
        Path to `.py` file which has an adjacent `.pyi` file.
        Typically use ``__file__``.

    Returns
    -------
    __getattr__, __dir__, __all__
        The same output as ``attach``.

    Raises
    ------
    ValueError
        If a stub file is not found for `filename`, or if the stubfile is formmated
        incorrectly (e.g. if it contains an relative import from outside of the module)
    """
    stubfile = (
        filename if filename.endswith("i") else f"{os.path.splitext(filename)[0]}.pyi"
    )

    if not os.path.exists(stubfile):
        raise ValueError(f"Cannot load imports from non-existent stub {stubfile!r}")

    with open(stubfile) as f:
        stub_node = ast.parse(f.read())

    visitor = _StubVisitor()
    visitor.visit(stub_node)
    return attach(package_name, visitor._submodules, visitor._submod_attrs)

import importlib
import sys
import types

import pytest

import lazy_loader as lazy


def test_lazy_import_basics():
    math = lazy.load("math")
    anything_not_real = lazy.load("anything_not_real")

    # Now test that accessing attributes does what it should
    assert math.sin(math.pi) == pytest.approx(0, 1e-6)
    # poor-mans pytest.raises for testing errors on attribute access
    try:
        anything_not_real.pi
        raise AssertionError()  # Should not get here
    except ModuleNotFoundError:
        pass
    assert isinstance(anything_not_real, lazy.DelayedImportErrorModule)
    # see if it changes for second access
    try:
        anything_not_real.pi
        raise AssertionError()  # Should not get here
    except ModuleNotFoundError:
        pass


def test_lazy_import_subpackages():
    with pytest.warns(RuntimeWarning):
        hp = lazy.load("html.parser")
    assert "html" in sys.modules
    assert type(sys.modules["html"]) == type(pytest)
    assert isinstance(hp, importlib.util._LazyModule)
    assert "html.parser" in sys.modules
    assert sys.modules["html.parser"] == hp


def test_lazy_import_impact_on_sys_modules():
    math = lazy.load("math")
    anything_not_real = lazy.load("anything_not_real")

    assert isinstance(math, types.ModuleType)
    assert "math" in sys.modules
    assert isinstance(anything_not_real, lazy.DelayedImportErrorModule)
    assert "anything_not_real" not in sys.modules

    # only do this if numpy is installed
    pytest.importorskip("numpy")
    np = lazy.load("numpy")
    assert isinstance(np, types.ModuleType)
    assert "numpy" in sys.modules

    np.pi  # trigger load of numpy

    assert isinstance(np, types.ModuleType)
    assert "numpy" in sys.modules


def test_lazy_import_nonbuiltins():
    np = lazy.load("numpy")
    sp = lazy.load("scipy")
    if not isinstance(np, lazy.DelayedImportErrorModule):
        assert np.sin(np.pi) == pytest.approx(0, 1e-6)
    if isinstance(sp, lazy.DelayedImportErrorModule):
        try:
            sp.pi
            raise AssertionError()
        except ModuleNotFoundError:
            pass


def test_lazy_attach():
    name = "mymod"
    submods = ["mysubmodule", "anothersubmodule"]
    myall = {"not_real_submod": ["some_var_or_func"]}

    locls = {
        "attach": lazy.attach,
        "name": name,
        "submods": submods,
        "myall": myall,
    }
    s = "__getattr__, __lazy_dir__, __all__ = attach(name, submods, myall)"

    exec(s, {}, locls)
    expected = {
        "attach": lazy.attach,
        "name": name,
        "submods": submods,
        "myall": myall,
        "__getattr__": None,
        "__lazy_dir__": None,
        "__all__": None,
    }
    assert locls.keys() == expected.keys()
    for k, v in expected.items():
        if v is not None:
            assert locls[k] == v


def test_attach_same_module_and_attr_name():
    from lazy_loader.tests import fake_pkg

    # Grab attribute twice, to ensure that importing it does not
    # override function by module
    assert isinstance(fake_pkg.some_func, types.FunctionType)
    assert isinstance(fake_pkg.some_func, types.FunctionType)

    # Ensure imports from submodule still work
    from lazy_loader.tests.fake_pkg.some_func import some_func

    assert isinstance(some_func, types.FunctionType)


FAKE_STUB = """
from . import rank
from ._gaussian import gaussian
from .edges import sobel, scharr, prewitt, roberts
"""


def test_stub_loading(tmp_path):
    stub = tmp_path / "stub.pyi"
    stub.write_text(FAKE_STUB)
    _get, _dir, _all = lazy.attach_stub("my_module", str(stub))
    expect = {"gaussian", "sobel", "scharr", "prewitt", "roberts", "rank"}
    assert set(_dir()) == set(_all) == expect


def test_stub_loading_parity():
    from lazy_loader.tests import fake_pkg

    from_stub = lazy.attach_stub(fake_pkg.__name__, fake_pkg.__file__)
    stub_getter, stub_dir, stub_all = from_stub
    assert stub_all == fake_pkg.__all__
    assert stub_dir() == fake_pkg.__lazy_dir__()
    assert stub_getter("some_func") == fake_pkg.some_func


def test_stub_loading_errors(tmp_path):
    stub = tmp_path / "stub.pyi"
    stub.write_text("from ..mod import func\n")

    with pytest.raises(ValueError, match="Only within-module imports are supported"):
        lazy.attach_stub("name", str(stub))

    with pytest.raises(ValueError, match="Cannot load imports from non-existent stub"):
        lazy.attach_stub("name", "not a file")

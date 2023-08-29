import lazy_loader as lazy

__getattr__, __lazy_dir__, __all__ = lazy.attach(
    __name__, submod_attrs={"some_func": ["some_func"]}
)

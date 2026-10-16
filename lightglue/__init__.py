from .lightglue import LightGlue


def __getattr__(name):
    if name=='SuperPoint':
        from .superpoint import SuperPoint
        return SuperPoint
    raise AttributeError(name)

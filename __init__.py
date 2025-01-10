try:
    from importlib.metadata import version
    __version__ = version("kozax")
except ImportError:
    __version__ = "unknown"
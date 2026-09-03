"""Isolated research harnesses sharing the repository experiment namespace."""

from pkgutil import extend_path


# ManiSoft-specific harnesses live below this directory while the formal DMC
# O2O implementation lives in the repository-level ``experiments`` package.
# Keep both roots visible when ManiSoft places its port root first on sys.path.
__path__ = extend_path(__path__, __name__)

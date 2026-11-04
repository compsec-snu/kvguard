"""Injected ``sitecustomize`` for the SGLang adapter (put this directory first on ``PYTHONPATH``).

Python imports ``sitecustomize`` during ``site`` initialization of every interpreter, including
the spawned scheduler subprocess. When ``KVGUARD_SGLANG_ATTACH`` is set this installs the
post-import hook of ``kvguard.adapters.sglang.hook``, then chains to the next ``sitecustomize``
further down ``sys.path`` so shadowing it has no effect. Attachment options travel as environment
variables the scheduler inherits and ``attach.attach_from_env`` reads.
"""

import importlib.machinery
import importlib.util
import os
import sys


def _install_kvguard_hook() -> None:
    if not os.environ.get("KVGUARD_SGLANG_ATTACH"):
        return
    try:
        from kvguard.adapters.sglang.hook import install
    except Exception as exc:
        sys.stderr.write(f"[kvguard-sglang] sitecustomize: cannot import the hook: {exc!r}\n")
        return
    install()


def _chain_next_sitecustomize() -> None:
    """Execute the ``sitecustomize`` that would have been found without this directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.origin == __file__ or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.stderr.write(f"[kvguard-sglang] sitecustomize: chained {spec.origin} failed: {exc!r}\n")


_install_kvguard_hook()
_chain_next_sitecustomize()

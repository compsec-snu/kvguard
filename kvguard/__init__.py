"""KVGuard: KV cache memory-safety checking for LLM serving systems.

``kvguard.core`` makes every safety decision, ``kvguard.gpu`` holds the GPU instrumentation, and
``kvguard.adapters`` translate serving-system state into the core's schema.
"""

__version__ = "0.0.1"

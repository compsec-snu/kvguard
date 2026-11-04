# KVGuard

**A KV cache memory-safety checker for LLM serving systems.**

LLM serving systems dynamically allocate and free KV cache as they serve
requests. This makes the KV cache inherit the classic memory safety bugs of a
heap allocator, such as use-after-free, reads of never-written memory, and
mismatched mappings, with no sanitizer to catch them. The backing GPU buffer is shared
across requests and never zeroed, so such a bug can let one request silently
read another request's KV, or uninitialized memory, while the serving system
returns a normal answer.

KVGuard attaches to an **unmodified** serving system and checks every KV access
against a shadow of what the serving system's own state says should be there,
reporting four violation classes: **P1** lifecycle, **P2** access, **P3**
mapping, **P4** initialization. See the paper for the design, threat model,
evaluation and related work.

> [!NOTE]
> **KVGuard is under active development.** This snapshot tracks the research
> tree, so its implementation and performance may differ from the paper.

## Supported serving systems

| Engine | Version | Attach                   |
| ------ | ------- | ------------------------ |
| vLLM   | 0.32.0  | `VLLM_PLUGINS=kvguard`   |
| SGLang | 0.5.22  | injected `sitecustomize` |

## Install

Install into the same environment as the serving system:

```bash
pip install -e /path/to/kvguard-public
```

## Quickstart

vLLM:

```bash
VLLM_PLUGINS=kvguard KVGUARD_VLLM_OUT=kvguard.json vllm serve <model>
```

SGLang:

```bash
INJECT=$(python -c "import os, kvguard.adapters.sglang as s; print(os.path.join(os.path.dirname(s.__file__), 'inject'))")
PYTHONPATH="$INJECT:$PYTHONPATH" KVGUARD_SGLANG_ATTACH=1 KVGUARD_SGLANG_OUT=kvguard.json \
  python -m sglang.launch_server --model <model>
```

KVGuard writes a JSON record to the `*_OUT` path with the violation counts and
reports, therefore a clean run reports nothing.

Detailed usage at: [`docs/usage.md`](docs/usage.md).

## Development

```bash
make setup   # build the tools venv (once, needs network)
make check   # unit tests + lint (hermetic: no GPU, no network)
make code    # optional GPU toolchain (interposer, validator PTX)
```

## License & citation

Apache-2.0 (see [`LICENSE`](LICENSE)).

```bibtex
@misc{cheong2026kvguard,
      title={KVGuard: A KV Cache Safety Checker for LLM Serving Systems},
      author={Jonguk Cheong and Myungbin Hwang and Eulgyu Kim and Byoungyoung Lee},
      year={2026},
      eprint={2611.06138},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2611.06138},
}
```

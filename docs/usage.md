# Usage

KVGuard attaches to an unmodified serving system at run time and writes a JSON
**record** of the run: the step count, pool geometry, violation counts by class,
and the first and last reports. A clean run reports nothing. Install KVGuard into
the same environment as the serving system (`pip install -e .`).

## vLLM 0.32.0

vLLM loads the `vllm.general_plugins` entry point in its engine-core (and worker)
processes. `VLLM_PLUGINS=kvguard` selects it, and it attaches at the first step.

```bash
VLLM_PLUGINS=kvguard KVGUARD_VLLM_OUT=kvguard.json vllm serve <model>
```

The same two variables work for the offline `LLM` API. The engine-core runs in a
subprocess by default and the plugin attaches there, so set the variables in the
environment that launches vLLM.

Environment variables (read in the engine-core process):

| Variable | Default | Meaning |
| --- | --- | --- |
| `KVGUARD_VLLM_OUT` | none | record path; without it nothing is written |
| `KVGUARD_VLLM_GPU_VALIDATE` | `0` | run the dispatch-time device validators |
| `KVGUARD_VLLM_TAGS` | `0` | run the per-slot tag tier |
| `KVGUARD_VLLM_TWINS` | `0` | substitute the instrumented twin kernels (see below) |
| `KVGUARD_VLLM_TABLES` | `device` | read tables from the device (`device`) or the manager mapping (`mapping`) |
| `KVGUARD_VLLM_AUDIT_EVERY` | `64` | full partition audit period, in steps |
| `KVGUARD_VLLM_CHECKER` | `inline` | run the checks `inline` or on a background `thread` |
| `KVGUARD_VLLM_FLUSH_EVERY` | `200` | how often the record is rewritten, in steps |

## SGLang 0.5.22

SGLang spawns its scheduler in a subprocess. KVGuard injects a `sitecustomize` so
it loads there: put the inject directory first on `PYTHONPATH` and set
`KVGUARD_SGLANG_ATTACH=1`.

```bash
INJECT=$(python -c "import os, kvguard.adapters.sglang as s; print(os.path.join(os.path.dirname(s.__file__), 'inject'))")
PYTHONPATH="$INJECT:$PYTHONPATH" \
KVGUARD_SGLANG_ATTACH=1 KVGUARD_SGLANG_OUT=kvguard.json \
  python -m sglang.launch_server --model <model>
```

Common variables (the scheduler subprocess inherits them):

| Variable | Default | Meaning |
| --- | --- | --- |
| `KVGUARD_SGLANG_ATTACH` | none | set to `1` to attach |
| `KVGUARD_SGLANG_OUT` | none | record path |
| `KVGUARD_SGLANG_GPU_VALIDATE` | `0` | run the dispatch-time device validators |
| `KVGUARD_SGLANG_TAGS` | `0` | run the per-slot tag tier |
| `KVGUARD_SGLANG_TWINS` | `0` | substitute the instrumented twin kernels |
| `KVGUARD_SGLANG_LIVE_NDJSON` | none | also stream each report as a JSON line, for a live reader |

The SGLang adapter has further flags for overlap-schedule semantics and opt-in
support of newer serving-system features (hybrid and recurrent-state caches,
sliding-window caches, speculative decoding, offload tiers). See the module
docstring at the top of `kvguard/adapters/sglang/attach.py` for the full list
and its support boundary.

## GPU instrumentation (optional)

The device validators (`*_GPU_VALIDATE=1`) and the tag tier (`*_TAGS=1`) need
only the validator PTX, built by `make code`. The twin kernels (`*_TWINS=1`) also
need the `LD_PRELOAD` interposer and a prebuilt twin cache: preload
`build/interposer/libkvg_interposer.so` and point `KVGUARD_INTERPOSER_CACHE` at
the twin cache in the serving system's environment. Everything in `make code`
builds without a GPU (it needs the CUDA toolkit, CMake, and Ninja); running the
instrumentation needs an `sm_80`+ GPU.

## Reading the record

The record is JSON. The fields that matter most:

- `attached`: whether KVGuard attached (false means the serving system ran unmodified).
- `steps`: serving-system steps observed.
- `report_kinds`, `host_report_kinds`, `device_report_kinds`: counts of each violation kind.
- `violations`: total violations.
- `reports_first`, `reports_last`: the first and last individual reports, each naming the pool, block, slot, and site.

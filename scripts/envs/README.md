# Environment recipes

Each directory here pins one environment that `scripts/setup-envs.sh` (and `make setup` / `make
envs`) builds with [uv](https://docs.astral.sh/uv/) under `$ENVS_DIR` (default `envs/`, which is
gitignored):

| Directory | Contents |
| --- | --- |
| `tools/` | `pytest`, `ruff`, `numpy` — used by `make check` (hermetic, offline after the first build) |
| `vllm-0.32.0/` | the vLLM 0.32.0 environment KVGuard's vLLM adapter targets |
| `sglang-0.5.22/` | the SGLang 0.5.22 environment KVGuard's SGLang adapter targets |

Each directory holds `requirements.in` (the direct pins), `requirements.lock` (the fully
resolved set), and `uv-args` (extra index/resolution flags). Rebuild the lock only with
`--lock`, and record why in the commit.

```bash
make setup                      # just the tools env (once, needs network)
make envs                       # the vLLM and SGLang environments (large)
bash scripts/setup-envs.sh vllm-0.32.0   # one environment by name
```

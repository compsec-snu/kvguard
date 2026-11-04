"""The GPU dispatch validators' host side (hermetic): the ledger's dirty set, the mirror words,
owner ids, and the kernels' source compiling to PTX when nvcc is present. The device behaviour is
covered by ``tests/gpu/test_validators.py`` (opt-in, needs the GPU)."""

import os
import shutil
import subprocess

import pytest
from kvguard.core import DenseBlockTable
from kvguard.core.ledger import BlockShadow
from kvguard.core.model import BlockState
from kvguard.gpu import validate as V

from tests.unit.conftest import BS, POOL, row

CU = os.path.join(
    os.path.dirname(__file__), "..", "..", "kvguard", "gpu", "validators", "validators.cu"
)


def test_pack_block_words():
    free = BlockShadow()
    assert V.pack_block(free, 0) == (0, 0, 0)
    assert V.pack_block(free, 0, reserved=True) == (V.RESERVED, 0, 0)
    private = BlockShadow(state=BlockState.PRIVATE, owner="A", filled=0b0111)
    assert V.pack_block(private, 5) == (V.ALLOC | (5 << V.OWNER_SHIFT), 0b0111, 0)
    pub = BlockShadow(state=BlockState.PUBLISHED, filled=0b0011, pending_publish=0b1100)
    st, fl, pe = V.pack_block(pub, 0)
    assert st == V.ALLOC | V.PUBLISHED | V.SHARED and (fl, pe) == (0b0011, 0b1100)
    # Wide blocks: the masks split into 32-bit words, low word first.
    wide = BlockShadow(
        state=BlockState.PRIVATE, owner="A", filled=(1 << 40) | 1, pending_publish=1 << 33
    )
    st, f0, f1, p0, p1 = V.pack_block(wide, 1, words=2)
    assert (f0, f1, p0, p1) == (1, 1 << 8, 0, 1 << 1)


def test_owner_ids_are_dense_and_stable():
    ids = V.OwnerIds()
    assert ids.id_of(None) == 0
    a, b = ids.id_of("A"), ids.id_of("B")
    assert (a, b) == (1, 2) and ids.id_of("A") == 1 and len(ids) == 2


def test_ledger_dirty_set_follows_every_shadow_mutation(world, ledger):
    assert ledger.drain_dirty(POOL) == set()
    world.alloc(3, 4)
    assert ledger.drain_dirty(POOL) == {3, 4}
    world.request("A", [3, 4], 6).observe()
    assert ledger.drain_dirty(POOL) == {3, 4}  # owner assigned by the snapshot
    world.observe()
    assert ledger.drain_dirty(POOL) == set()  # an unchanged snapshot dirties nothing
    world.dispatch("w", [row("A", writes=range(0, 6))], DenseBlockTable([[3, 4]], [2]))
    assert ledger.drain_dirty(POOL) == {3, 4}  # filled bits
    assert ledger.block(POOL, 3).filled == (1 << BS) - 1
    world.complete("w")
    assert ledger.drain_dirty(POOL) == set()  # in-flight counts are not mirrored
    world.drop_request("A").release(3, 4).observe()
    assert ledger.drain_dirty(POOL) == {3, 4}


def test_kernel_names_cover_every_representation():
    # One validator per consumed representation (plus the TensorRT-LLM offsets table) and the
    # mirror update; the PTX test below checks each name is an entry point.
    assert V.KERNELS == (
        "kvg_validate_dense",
        "kvg_validate_dense_offsets",
        "kvg_validate_csr",
        "kvg_validate_token_csr",  # the per-token page list at any page size, against the mapping
        "kvg_validate_token_table",
        "kvg_validate_slot_map",
        "kvg_mirror_update",
        # The tag tier: the stamp, one check per representation, the ranges a transfer engine
        # reads, and the maintenance records.
        "kvg_tag_stamp",
        "kvg_tag_check_dense",
        "kvg_tag_check_dense_offsets",
        "kvg_tag_check_csr",
        "kvg_tag_check_token_table",
        "kvg_tag_check_ranges",
        "kvg_tag_update",
        "kvg_tag_update_slots",
    )
    assert hasattr(V.Validators, "validate_dense_offsets")


def test_default_ptx_path_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KVGUARD_VALIDATORS_PTX", str(tmp_path / "v.ptx"))
    assert V.default_ptx_path() == tmp_path / "v.ptx"
    monkeypatch.delenv("KVGUARD_VALIDATORS_PTX")
    assert V.default_ptx_path().name == "validators.ptx"


def _nvcc():
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    for cand in (os.path.join(cuda, "bin", "nvcc"), shutil.which("nvcc")):
        if cand and os.path.exists(cand):
            return cand
    return None


@pytest.mark.skipif(_nvcc() is None, reason="nvcc not installed")
def test_validator_source_compiles_to_ptx_with_every_kernel(tmp_path):
    out = tmp_path / "validators.ptx"
    r = subprocess.run(
        [_nvcc(), "-arch=sm_80", "-O3", "-ptx", CU, "-o", str(out)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    for name in V.KERNELS:
        assert f".visible .entry {name}(" in text
    ptxas = os.path.join(os.path.dirname(_nvcc()), "ptxas")
    if os.path.exists(ptxas):
        r = subprocess.run(
            [ptxas, "-arch=sm_86", str(out), "-o", "/dev/null"], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr

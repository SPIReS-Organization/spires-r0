import os
import shutil

from spires_r0.cache import R0Cache
from spires_r0.core import R0_BUILD_SIGNATURE_ATTR
from spires_r0.raster import build_raster_r0
from tests.conftest import make_raster_scene

SNOWFREE = {"blue": 0.05, "green": 0.08, "red": 0.10, "nir": 0.40, "swir1": 0.25, "swir2": 0.15}


def _r0(tmp_path):
    d = tmp_path / "landsat" / "A"
    d.mkdir(parents=True)
    p = tmp_path / "src.nc"
    ds = build_raster_r0([d], sensor="landsat", r0_path=p,
                         open_scene=lambda _: make_raster_scene(SNOWFREE), prefetch=False)
    return p, ds.attrs[R0_BUILD_SIGNATURE_ATTR]


def test_put_get_copies_and_validates(tmp_path):
    src, key = _r0(tmp_path)
    cache = R0Cache(tmp_path / "cache")
    cache.put(key, src, meta={})
    dest = tmp_path / "out" / "r0.nc"
    assert cache.get(key, dest) and dest.exists()
    assert os.stat(dest).st_ino != os.stat(cache.root / key / "r0.nc").st_ino
    assert (cache.root / key / "meta.json").exists()


def test_get_misses_on_unknown_corrupt_and_wrong_signature(tmp_path):
    src, key = _r0(tmp_path)
    cache = R0Cache(tmp_path / "cache")
    assert cache.get("nope", tmp_path / "x.nc") is False
    (cache.root / "bad").mkdir()
    (cache.root / "bad" / "r0.nc").write_bytes(b"junk")
    assert cache.get("bad", tmp_path / "y.nc") is False
    cache.put("mismatch", src, meta={})  # file's signature != key
    assert cache.get("mismatch", tmp_path / "z.nc") is False


def test_put_is_atomic(tmp_path):
    src, key = _r0(tmp_path)
    cache = R0Cache(tmp_path / "cache")
    cache.put(key, src, meta={})
    assert sorted(p.name for p in cache.root.iterdir()) == [key]


def test_driver_cache_hit_skips_loading(tmp_path):
    d = tmp_path / "landsat" / "A"
    d.mkdir(parents=True)
    n = {"c": 0}

    def opener(_):
        n["c"] += 1
        return make_raster_scene(SNOWFREE)

    root = tmp_path / "cache"
    build_raster_r0([d], sensor="landsat", r0_path=tmp_path / "1.nc", cache_root=root,
                    open_scene=opener, prefetch=False)
    build_raster_r0([d], sensor="landsat", r0_path=tmp_path / "2.nc", cache_root=root,
                    open_scene=opener, prefetch=False)
    assert n["c"] == 1 and (tmp_path / "2.nc").exists()


def test_cache_put_failure_does_not_fail_build(tmp_path, monkeypatch, caplog):
    d = tmp_path / "landsat" / "A"
    d.mkdir(parents=True)

    def boom(self, *a, **k):
        raise OSError("simulated cache failure")

    monkeypatch.setattr(R0Cache, "put", boom)
    r0_path = tmp_path / "r0.nc"
    with caplog.at_level("WARNING"):
        ds = build_raster_r0([d], sensor="landsat", r0_path=r0_path, cache_root=tmp_path / "cache",
                             open_scene=lambda _: make_raster_scene(SNOWFREE), prefetch=False)
    assert ds is not None and "r0_reflectance" in ds
    assert r0_path.exists()
    assert "R0 cache put failed" in caplog.text


def test_get_unlinks_tmp_when_copy_fails(tmp_path, monkeypatch):
    src, key = _r0(tmp_path)
    cache = R0Cache(tmp_path / "cache")
    cache.put(key, src, meta={})
    real_copy = shutil.copyfile

    def partial_then_fail(s, dst, *a, **k):
        real_copy(s, dst)
        raise OSError("simulated short write")

    monkeypatch.setattr(shutil, "copyfile", partial_then_fail)
    out = tmp_path / "out" / "r0.nc"
    assert cache.get(key, out) is False
    assert not out.exists()
    assert list((tmp_path / "out").iterdir()) == []


def test_put_tolerates_concurrent_winner(tmp_path, monkeypatch):
    src, key = _r0(tmp_path)
    cache = R0Cache(tmp_path / "cache")
    real_rename = os.rename

    def racing_rename(a, b):
        # A sibling writer publishes the same key after our exists()/rmtree() check, before our rename.
        (cache.root / key).mkdir()
        (cache.root / key / "sibling.txt").write_text("winner")
        return real_rename(a, b)

    monkeypatch.setattr(os, "rename", racing_rename)
    assert cache.put(key, src, meta={}) == cache.root / key
    assert sorted(p.name for p in cache.root.iterdir()) == [key]
    assert (cache.root / key / "sibling.txt").exists()

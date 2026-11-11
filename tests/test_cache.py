import os

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

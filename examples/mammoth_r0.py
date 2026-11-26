"""Build an HLS L30 R0 (snow-free background) over Mammoth Mountain from spires_io-materialised scenes.

    python examples/mammoth_r0.py scenes/hls_l30 mammoth_hlsl30_r0.nc [watermask_dir]

`watermask_dir` holds landWater2020 tiles; without it lakes get no background reflectance.
"""
import logging
import sys
from pathlib import Path

from spires_r0 import build_raster_r0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
root, out = Path(sys.argv[1]), Path(sys.argv[2])
water = Path(sys.argv[3]) if len(sys.argv) > 3 else None
dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".partial")]
print(build_raster_r0(dirs, sensor="hls_l30", r0_path=out, water_mask_dir=water, cache_root=out.parent / ".r0_cache"))

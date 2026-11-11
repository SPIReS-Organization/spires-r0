"""Build an HLS L30 R0 (snow-free background) over Mammoth Mountain from spires_io-materialised scenes.

    python examples/mammoth_r0.py /data/spires/scenes/hls_l30 /data/spires/r0/mammoth_hlsl30_r0.nc
"""
import logging
import sys
from pathlib import Path

from spires_r0 import build_raster_r0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
root, out = Path(sys.argv[1]), Path(sys.argv[2])
dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".partial")]
print(build_raster_r0(dirs, sensor="hls_l30", r0_path=out, cache_root=out.parent / ".r0_cache"))

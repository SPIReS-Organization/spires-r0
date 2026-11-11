"""Build-signature-keyed R0 cache. Hit = copy (never link) after re-validation; any problem = miss."""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

from spires_r0.core import load_existing_r0_if_valid

log = logging.getLogger(__name__)
ARTIFACT, META = "r0.nc", "meta.json"


class R0Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, dest: str | Path) -> bool:
        src = self.root / key / ARTIFACT
        try:
            if not src.is_file():
                return False
            existing = load_existing_r0_if_valid(src, expected_band_count=None, max_scene_index=None,
                                                 expected_source_inventory_digest=None,
                                                 expected_build_signature=key)
            if existing is None:
                return False
            existing.close()  # release the handle before copying
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.parent / f".{dest.name}.{uuid.uuid4().hex}.tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest)
            return True
        except Exception as exc:  # noqa: BLE001 - a cache may save time, never fail a build
            log.warning("R0 cache miss for %s: %s", key, exc)
            return False

    def put(self, key: str, src: str | Path, *, meta: dict) -> Path:
        final = self.root / key
        partial = self.root / f".{key}.{uuid.uuid4().hex}.partial"
        partial.mkdir(parents=True)
        try:
            shutil.copyfile(src, partial / ARTIFACT)
            (partial / META).write_text(json.dumps({"key": key, **meta}, indent=2, default=str))
            if final.exists():
                shutil.rmtree(final)
            os.rename(partial, final)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return final

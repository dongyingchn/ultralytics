from __future__ import annotations
import random
from typing import Dict, Optional, Tuple, Union

ROITuple = Tuple[int, int, int, int]
WH = Tuple[int, int]
CropWH = Tuple[int, int]


class BaseROIPolicy:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def get_roi(self, im_size: WH, meta: Optional[dict] = None) -> Optional[ROITuple]:
        raise NotImplementedError


class StaticROI(BaseROIPolicy):
    def __init__(self, roi: ROITuple):
        super().__init__(seed=None)
        self.roi = tuple(int(v) for v in roi)

    def get_roi(self, im_size: WH, meta: Optional[dict] = None) -> ROITuple:
        return self.roi


class RandomWindowROI(BaseROIPolicy):
    def __init__(
        self,
        rules: Optional[Dict[WH, CropWH]] = None,
        aspect_ratio: float = 2.5,
        seed: Optional[int] = None,
        prefer_full_width: bool = False,
    ):
        super().__init__(seed=seed)
        self.rules = rules or {}
        self.aspect_ratio = float(aspect_ratio)
        self.prefer_full_width = bool(prefer_full_width)

    def _decide_crop_size(self, w: int, h: int) -> CropWH:
        if (w, h) in self.rules:
            cw, ch = self.rules[(w, h)]
            return int(cw), int(ch)

        if self.prefer_full_width:
            crop_w = w
            crop_h = int(round(w / self.aspect_ratio))
            if crop_h > h:
                crop_h = h
                crop_w = int(round(h * self.aspect_ratio))
        else:
            crop_h = min(h, int(round(w / self.aspect_ratio)))
            crop_w = int(round(crop_h * self.aspect_ratio))
            if crop_w > w:
                crop_w = w
                crop_h = int(round(w / self.aspect_ratio))
        return max(1, crop_w), max(1, crop_h)

    def get_roi(self, im_size: WH, meta: Optional[dict] = None) -> ROITuple:
        w, h = int(im_size[0]), int(im_size[1])
        crop_w, crop_h = self._decide_crop_size(w, h)
        max_x = max(0, w - crop_w)
        max_y = max(0, h - crop_h)
        x1 = self._rng.randint(0, max_x) if max_x > 0 else 0
        y1 = self._rng.randint(0, max_y) if max_y > 0 else 0
        return int(x1), int(y1), int(x1 + crop_w), int(y1 + crop_h)


class MixedROI(BaseROIPolicy):
    def __init__(self, static_map: Dict[WH, ROITuple], fallback: BaseROIPolicy):
        super().__init__(seed=None)
        self.static_map = {(int(w), int(h)): tuple(int(v) for v in roi) for (w, h), roi in static_map.items()}
        self.fallback = fallback

    def get_roi(self, im_size: WH, meta: Optional[dict] = None) -> ROITuple:
        w, h = int(im_size[0]), int(im_size[1])
        if (w, h) in self.static_map:
            return self.static_map[(w, h)]
        return self.fallback.get_roi(im_size, meta)


def _parse_wh_key(k: Union[str, WH, CropWH]) -> WH:
    if isinstance(k, (list, tuple)) and len(k) == 2:
        return int(k[0]), int(k[1])
    if isinstance(k, str):
        s = k.lower().replace(" ", "")
        if "x" in s:
            w, h = s.split("x", 1)
            return int(w), int(h)
    raise ValueError(f"Invalid resolution key: {k}. Expect 'WxH' or [W, H].")


def _parse_roi(v: Union[list, tuple]) -> ROITuple:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError(f"Invalid ROI: {v}. Expect [x1,y1,x2,y2].")
    return tuple(int(x) for x in v)


def _parse_rules(d: Dict[Union[str, list, tuple], Union[list, tuple]]) -> Dict[WH, CropWH]:
    out: Dict[WH, CropWH] = {}
    for k, v in (d or {}).items():
        wh = _parse_wh_key(k)
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"Invalid crop size for {k}: {v}. Expect [crop_w,crop_h].")
        out[wh] = (int(v[0]), int(v[1]))
    return out


def build_roi_policy_from_config(cfg: dict) -> BaseROIPolicy:
    t = str(cfg.get("type", "static")).lower()

    if t == "static":
        roi = cfg.get("roi", None)
        if roi is None:
            raise ValueError("roi_policy.type='static' requires field 'roi'")
        return StaticROI(_parse_roi(roi))

    if t == "random_window":
        rules = _parse_rules(cfg.get("rules", {}) or {})
        aspect_ratio = float(cfg.get("aspect_ratio", 2.5))
        seed = cfg.get("seed", None)
        prefer_full_width = bool(cfg.get("prefer_full_width", False))
        return RandomWindowROI(rules=rules, aspect_ratio=aspect_ratio, seed=seed, prefer_full_width=prefer_full_width)

    if t == "mixed":
        static_map_cfg = cfg.get("static", {})
        if not isinstance(static_map_cfg, dict) or not static_map_cfg:
            raise ValueError("roi_policy.type='mixed' requires non-empty 'static' mapping")
        static_map = {_parse_wh_key(k): _parse_roi(v) for k, v in static_map_cfg.items()}
        random_cfg = cfg.get("random", None)
        if not isinstance(random_cfg, dict):
            raise ValueError("roi_policy.type='mixed' requires 'random' sub-config to fallback")
        fallback = build_roi_policy_from_config(random_cfg)
        return MixedROI(static_map=static_map, fallback=fallback)

    raise ValueError(f"Unsupported roi_policy.type='{t}'")
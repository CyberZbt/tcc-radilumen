# Trecho essencial — src/preprocess.py
def preprocess_array(img, config=None):
    cfg = config or PreprocessConfig()
    gray = to_uint8(
        img,
        mode=cfg.normalize,
        p_low=cfg.percentile_low,
        p_high=cfg.percentile_high,
    )
    if cfg.clahe:
        gray = apply_clahe(gray, cfg.clip_limit, cfg.tile_grid_size)
    if cfg.output_bgr:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

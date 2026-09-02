"""Shared Euler integration core: dP/5 + foreground mask → Euler → labels.

Matches Cellpose official compute_masks internals:
  - steps_interp: exact Cellpose coordinate normalization and flow scaling
    (pt = pixel/(N-1)*2-1, flow *= 2/(N-1), denormalize by N-1 + trunc)
  - get_masks_torch: rpad=20 histogram, 5×5 max-pool seeds, per-pixel seed
    processing with amax overlap resolution, 5-iteration 3×3 dilation with
    h_slc>2 filtering, max_size_fraction removal AFTER foreground projection

Reference: Cellpose dynamics.py steps_interp, follow_flows, get_masks_torch.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Euler integration — exact Cellpose steps_interp formulas
# ═══════════════════════════════════════════════════════════════════════════════
#
# Cellpose normalization (2D):
#   shape_orig = (H, W)
#   pt = pixel / (N-1) * 2 - 1           (maps [0, N-1] → [-1, 1])
#   dP_norm = dP * 2 / (N-1)             (pixel-space flow → [-1,1]-space)
#   grid_sample(align_corners=False): [-1,1] → pixel = ((c+1)*N - 1)/2
# After Euler, denormalize in follow_flows → steps_interp:
#   pixel = (pt + 1) * 0.5 * (N - 1)      (uses N-1, Cellpose exact)
#   Then int truncation + clamp to [0, N-1]

def euler_integrate_to_sinks(
    dy: np.ndarray,             # (H, W) float32, raw model dy
    dx: np.ndarray,             # (H, W) float32, raw model dx
    fg_mask: np.ndarray,        # (H, W) bool, foreground mask
    niter: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Run Euler integration with exact Cellpose normalization.

    Returns:
        (ys_all, xs_all, sink_lin, W, H):
          sink_lin: (N,) int32 convergence linear indices in [0, H*W-1]
    """
    H, W = dy.shape

    fg_float = fg_mask.astype(np.float32)
    dP_y = dy / 5.0 * fg_float
    dP_x = dx / 5.0 * fg_float

    ys_all, xs_all = np.nonzero(fg_float > 0)
    n_pts = len(ys_all)
    if n_pts == 0:
        return ys_all, xs_all, np.array([], dtype=np.int32), W, H

    ys_all_f = ys_all.astype(np.float32)
    xs_all_f = xs_all.astype(np.float32)

    # ── Cellpose normalization ──
    # pt = pixel / (N-1) * 2 - 1
    inv_Hm1 = 1.0 / max(H - 1, 1.0)
    inv_Wm1 = 1.0 / max(W - 1, 1.0)
    flow_scale_y = 2.0 * inv_Hm1  # 2/(H-1)
    flow_scale_x = 2.0 * inv_Wm1  # 2/(W-1)

    dP_norm_y = dP_y * flow_scale_y
    dP_norm_x = dP_x * flow_scale_x

    pt_y = ys_all_f * inv_Hm1 * 2.0 - 1.0
    pt_x = xs_all_f * inv_Wm1 * 2.0 - 1.0

    # ── Euler integration with per-neighbor zero padding ──
    # grid_sample(align_corners=False, padding_mode='zeros'):
    #   - Convert [-1,1] coord to pixel coord (unclamped)
    #   - For each of 4 bilinear neighbors, check if within [0,N-1]
    #   - Out-of-bounds neighbor contributes 0; in-bounds uses bilinear weight
    for _ in range(niter):
        imgY = ((pt_y + 1.0) * H - 1.0) * 0.5  # unclamped
        imgX = ((pt_x + 1.0) * W - 1.0) * 0.5

        y0 = np.floor(imgY).astype(np.int32); y1 = y0 + 1
        x0 = np.floor(imgX).astype(np.int32); x1 = x0 + 1

        wy = imgY - y0.astype(np.float32)
        wx = imgX - x0.astype(np.float32)

        # per-neighbor validity (zero padding for out-of-bounds)
        v00 = (y0 >= 0) & (y0 < H) & (x0 >= 0) & (x0 < W)
        v01 = (y0 >= 0) & (y0 < H) & (x1 >= 0) & (x1 < W)
        v10 = (y1 >= 0) & (y1 < H) & (x0 >= 0) & (x0 < W)
        v11 = (y1 >= 0) & (y1 < H) & (x1 >= 0) & (x1 < W)

        # safe indices (clamped only for memory access, not for sampling logic)
        y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
        x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)

        dy_i = (np.where(v00, (1.0 - wy) * (1.0 - wx) * dP_norm_y[y0c, x0c], 0.0) +
                np.where(v01, (1.0 - wy) *        wx  * dP_norm_y[y0c, x1c], 0.0) +
                np.where(v10,        wy  * (1.0 - wx) * dP_norm_y[y1c, x0c], 0.0) +
                np.where(v11,        wy  *        wx  * dP_norm_y[y1c, x1c], 0.0))
        dx_i = (np.where(v00, (1.0 - wy) * (1.0 - wx) * dP_norm_x[y0c, x0c], 0.0) +
                np.where(v01, (1.0 - wy) *        wx  * dP_norm_x[y0c, x1c], 0.0) +
                np.where(v10,        wy  * (1.0 - wx) * dP_norm_x[y1c, x0c], 0.0) +
                np.where(v11,        wy  *        wx  * dP_norm_x[y1c, x1c], 0.0))

        pt_y = np.clip(pt_y + dy_i, -1.0, 1.0)
        pt_x = np.clip(pt_x + dx_i, -1.0, 1.0)

    # ── Denormalize: (pt+1)/2 * (N-1) + truncation (Cellpose exact) ──
    final_y = (pt_y + 1.0) * 0.5 * (H - 1)
    final_x = (pt_x + 1.0) * 0.5 * (W - 1)
    yi = np.clip(final_y.astype(np.int32), 0, H - 1)
    xi = np.clip(final_x.astype(np.int32), 0, W - 1)
    sink_lin = yi * W + xi

    return ys_all, xs_all, sink_lin, W, H


# ═══════════════════════════════════════════════════════════════════════════════
# Mask clustering — exact Cellpose get_masks_torch algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def _max_pool_2d(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """2D max pool with stride=1, same-size output via zero padding."""
    try:
        from scipy.ndimage import maximum_filter
        return maximum_filter(data, size=kernel_size, mode='constant', cval=0.0)
    except ImportError:
        pass
    H, W = data.shape
    pad = kernel_size // 2
    padded = np.pad(data.astype(np.float64), pad, mode='constant', constant_values=0)
    result = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            result[y, x] = np.max(padded[y:y + kernel_size, x:x + kernel_size])
    return result


def get_masks_torch(
    sink_lin: np.ndarray,      # (N,) int32 convergence linear indices
    H: int, W: int,
    rpad: int = 20,
) -> np.ndarray:
    """Convert convergence sinks to instance labels.

    Exact Cellpose get_masks_torch:
      1. rpad=20 padded histogram
      2. 5×5 max-pool → seed pixels (h > 10, local maximum)
      3. Per-pixel seeds sorted by peak height ascending
      4. Each seed: 11×11 window, 5×3×3 dilation with h_slc>2 filter
      5. Scatter amax (larger label = larger peak wins)
      6. Crop to original size

    Note: max_size_fraction removal is NOT done here; it is applied
    after projection to foreground pixels (as in official Cellpose).

    Returns:
        labels: (H, W) int32 label map on the ORIGINAL (unpadded) image.
    """
    if len(sink_lin) == 0:
        return np.zeros((H, W), dtype=np.int32)

    H_pad = H + 2 * rpad
    W_pad = W + 2 * rpad

    yi_conv = sink_lin // W + rpad
    xi_conv = sink_lin % W + rpad

    h = np.zeros((H_pad, W_pad), dtype=np.float64)
    valid = (yi_conv >= 0) & (yi_conv < H_pad) & (xi_conv >= 0) & (xi_conv < W_pad)
    np.add.at(h, (np.clip(yi_conv, 0, H_pad - 1),
                  np.clip(xi_conv, 0, W_pad - 1)), 1.0)

    # 5×5 max pool → seed detection
    h_max = _max_pool_2d(h, 5)
    seed_mask = (np.abs(h - h_max) < 1e-10) & (h > 10.0)
    seed_ys, seed_xs = np.nonzero(seed_mask)
    n_seeds = len(seed_ys)
    if n_seeds == 0:
        return np.zeros((H, W), dtype=np.int32)

    # Per-pixel seeds: get peak heights, sort ascending
    seed_heights = h[seed_ys, seed_xs]
    order = np.argsort(seed_heights)  # ascending

    # ── Process each seed pixel independently ──
    mask = np.zeros((H_pad, W_pad), dtype=np.int32)

    for rank, idx in enumerate(order):
        cy = seed_ys[idx]
        cx = seed_xs[idx]
        label = rank + 1  # larger label = larger peak → wins in amax

        y0 = max(0, cy - 5)
        y1 = min(H_pad, cy + 6)
        x0 = max(0, cx - 5)
        x1 = min(W_pad, cx + 6)
        winH = y1 - y0
        winW = x1 - x0
        if winH <= 0 or winW <= 0:
            continue

        h_slc = h[y0:y1, x0:x1].copy()

        seed_dilated = np.zeros((winH, winW), dtype=np.float64)
        rel_y = cy - y0
        rel_x = cx - x0
        if 0 <= rel_y < winH and 0 <= rel_x < winW:
            seed_dilated[rel_y, rel_x] = 1.0

        for _ in range(5):
            seed_dilated = _max_pool_2d(seed_dilated, 3)
            seed_dilated *= (h_slc > 2.0)

        # Scatter amax into global mask
        active = seed_dilated > 0
        if np.any(active):
            region = mask[y0:y1, x0:x1]
            replacement = np.where(active, label, 0)
            np.maximum(region, replacement.astype(np.int32), out=region)

    # Crop to original size
    return mask[rpad:rpad + H, rpad:rpad + W].astype(np.int32)


def euler_integrate_to_labels(
    dy: np.ndarray,
    dx: np.ndarray,
    fg_mask: np.ndarray,
    niter: int = 200,
) -> np.ndarray:
    """Full pipeline: Euler + get_masks_torch clustering + projection.

    max_size_fraction removal is NOT done here — callers apply it
    after this function returns (on the per-pixel label map).

    Returns:
        labels: (H, W) int32 instance label map.
    """
    ys_all, xs_all, sink_lin, W, H = euler_integrate_to_sinks(dy, dx, fg_mask, niter)

    if len(sink_lin) == 0:
        return np.zeros(fg_mask.shape, dtype=np.int32)

    mask_labels = get_masks_torch(sink_lin, H, W, rpad=20)

    # Project: each foreground pixel gets label from its convergence sink
    results = np.zeros((H, W), dtype=np.int32)
    yi_sink = sink_lin // W
    xi_sink = sink_lin % W
    results[ys_all.astype(np.int32), xs_all.astype(np.int32)] = \
        mask_labels[yi_sink, xi_sink]
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing filters
# ═══════════════════════════════════════════════════════════════════════════════

def remove_large_masks(
    labels: np.ndarray,
    max_size_fraction: float,
) -> np.ndarray:
    """Remove masks exceeding max_size_fraction of total image area.

    Applied AFTER projection to foreground pixels, matching Cellpose's
    ordering: projection → fastremap.mask → fastremap.renumber.
    """
    if max_size_fraction <= 0 or labels.max() == 0:
        return labels
    H, W = labels.shape
    max_px = H * W * max_size_fraction
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        if (labels == lbl).sum() > max_px:
            labels[labels == lbl] = 0
    return labels


def fill_holes_labels(labels: np.ndarray) -> np.ndarray:
    """Fill holes in each instance mask (Cellpose fill_voids.fill equivalent).

    Uses scipy.ndimage.binary_fill_holes when available, otherwise no-op.
    """
    if labels.max() == 0:
        return labels
    try:
        from scipy.ndimage import binary_fill_holes
        for lbl in np.unique(labels):
            if lbl == 0:
                continue
            m = (labels == lbl)
            labels[binary_fill_holes(m) & ~m] = lbl
    except ImportError:
        pass
    return labels


def filter_by_size_and_boundary(
    labels: np.ndarray,
    min_size: int,
    max_size_fraction: float,
    boundary_particle_policy: str = 'include',
    edge_touch_margin_px: float = 1.0,
) -> np.ndarray:
    """Remove labels that fail size or (when policy=='exclude') boundary checks.

    Boundary check uses contour bounding-rect semantics aligned with
    C++ FlowDynamics::buildParticles.
    """
    if labels.max() == 0:
        return labels

    H, W = labels.shape
    max_area = H * W * max_size_fraction

    max_label = int(labels.max())
    for lbl in range(1, max_label + 1):
        m = (labels == lbl)
        area_cnt = int(m.sum())
        if area_cnt < min_size:
            labels[m] = 0
            continue
        if max_size_fraction > 0 and area_cnt > max_area:
            labels[m] = 0
            continue

        if boundary_particle_policy == 'exclude':
            import cv2
            m_u8 = m.astype(np.uint8)
            conts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if conts:
                best = max(conts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(best)
                touches_boundary = (
                    x <= edge_touch_margin_px or
                    y <= edge_touch_margin_px or
                    x + w >= W - 1 - edge_touch_margin_px or
                    y + h >= H - 1 - edge_touch_margin_px
                )
                if touches_boundary:
                    labels[m] = 0

    return labels


def filter_by_mean_logit(
    labels: np.ndarray,
    cellprob_logit: np.ndarray,
    threshold_logit: float,
) -> np.ndarray:
    """Remove labels whose mean cellprob logit is below threshold."""
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        m = (labels == lbl)
        if cellprob_logit[m].mean() < threshold_logit:
            labels[m] = 0
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# masks_to_flows — Python equivalent of C++ FlowDynamics::masksToFlows()
# ═══════════════════════════════════════════════════════════════════════════════

def _numpy_round_to_int(value: float) -> int:
    """numpy rounding: round half to even."""
    floor_v = int(np.floor(value))
    fraction = value - floor_v
    if fraction < 0.5:
        return floor_v
    if fraction > 0.5:
        return floor_v + 1
    return floor_v if (floor_v % 2 == 0) else floor_v + 1


def masks_to_flows(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate (dy, dx) flow fields from instance labels using heat diffusion.

    Equivalent to C++ FlowDynamics::masksToFlows() and Cellpose masks_to_flows_gpu.

    Args:
        labels: (H, W) int32 instance label map, 0=background, 1..N=instances.

    Returns:
        (dy, dx): each (H, W) float64, L2-normalized flow vectors.
    """
    H, W = labels.shape
    dy_all = np.zeros((H, W), dtype=np.float64)
    dx_all = np.zeros((H, W), dtype=np.float64)

    ids = [v for v in np.unique(labels) if v > 0]
    if not ids:
        return dy_all, dx_all

    n_labels = int(max(ids))

    # ── Per-label statistics ──
    areas = np.zeros(n_labels + 1, dtype=np.int32)
    min_y = np.full(n_labels + 1, H, dtype=np.int32)
    min_x = np.full(n_labels + 1, W, dtype=np.int32)
    max_y = np.full(n_labels + 1, -1, dtype=np.int32)
    max_x = np.full(n_labels + 1, -1, dtype=np.int32)
    sum_y = np.zeros(n_labels + 1, dtype=np.float64)
    sum_x = np.zeros(n_labels + 1, dtype=np.float64)

    foreground: list[tuple[int, int, int]] = []  # (py, px, label) in padded coords

    for y in range(H):
        for x in range(W):
            lbl = labels[y, x]
            if lbl <= 0 or lbl > n_labels:
                continue
            areas[lbl] += 1
            if y < min_y[lbl]:
                min_y[lbl] = y
            if x < min_x[lbl]:
                min_x[lbl] = x
            if y > max_y[lbl]:
                max_y[lbl] = y
            if x > max_x[lbl]:
                max_x[lbl] = x
            sum_y[lbl] += y
            sum_x[lbl] += x
            foreground.append((y + 1, x + 1, lbl))  # padded coords

    if not foreground:
        return dy_all, dx_all

    # ── Compute centers ──
    max_extent = 0
    centers: list[tuple[int, int]] = []  # (cy, cx) in padded coords

    for lbl in range(1, n_labels + 1):
        if areas[lbl] <= 0:
            continue
        height = max_y[lbl] - min_y[lbl] + 1
        width = max_x[lbl] - min_x[lbl] + 1
        max_extent = max(max_extent, height + width + 2)

        local_mean_y = (sum_y[lbl] - min_y[lbl] * areas[lbl]) / areas[lbl]
        local_mean_x = (sum_x[lbl] - min_x[lbl] * areas[lbl]) / areas[lbl]
        target_local_y = _numpy_round_to_int(local_mean_y)
        target_local_x = _numpy_round_to_int(local_mean_x)

        center_y = min_y[lbl] + target_local_y
        center_x = min_x[lbl] + target_local_x

        if (center_y < min_y[lbl] or center_y > max_y[lbl]
                or center_x < min_x[lbl] or center_x > max_x[lbl]
                or labels[center_y, center_x] != lbl):
            best_y, best_x = min_y[lbl], min_x[lbl]
            best_dist = float('inf')
            for y in range(min_y[lbl], max_y[lbl] + 1):
                for x in range(min_x[lbl], max_x[lbl] + 1):
                    if labels[y, x] != lbl:
                        continue
                    ddy = y - min_y[lbl] - target_local_y
                    ddx = x - min_x[lbl] - target_local_x
                    dist = ddy * ddy + ddx * ddx
                    if dist < best_dist:
                        best_dist = dist
                        best_y, best_x = y, x
            center_y, center_x = best_y, best_x

        centers.append((center_y + 1, center_x + 1))  # padded coords

    if not centers or max_extent <= 0:
        return dy_all, dx_all

    # ── Heat diffusion ──
    n_iter = 2 * max_extent
    padded_labels = np.pad(labels.astype(np.int32), 1, mode='constant', constant_values=0)
    temperature = np.zeros((H + 2, W + 2), dtype=np.float64)
    next_temperature = np.zeros((H + 2, W + 2), dtype=np.float64)

    neighbor_offsets = [
        (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    for _ in range(n_iter):
        for cy, cx in centers:
            temperature[cy, cx] += 1.0

        for py, px, lbl in foreground:
            total = 0.0
            for dy_off, dx_off in neighbor_offsets:
                yy, xx = py + dy_off, px + dx_off
                if padded_labels[yy, xx] == lbl:
                    total += temperature[yy, xx]
            next_temperature[py, px] = total / 9.0

        temperature, next_temperature = next_temperature, temperature

    # ── Gradient + L2 normalization ──
    for py, px, lbl in foreground:
        dyy = temperature[py + 1, px] - temperature[py - 1, px]
        dxx = temperature[py, px + 1] - temperature[py, px - 1]
        norm = np.sqrt(dyy * dyy + dxx * dxx) + 1e-60
        dy_all[py - 1, px - 1] = dyy / norm
        dx_all[py - 1, px - 1] = dxx / norm

    return dy_all, dx_all


def remove_bad_flow_masks(
    labels: np.ndarray,
    raw_dy: np.ndarray,
    raw_dx: np.ndarray,
    flow_threshold: float,
) -> np.ndarray:
    """Remove masks whose mean flow error exceeds threshold.

    Equivalent to C++ FlowDynamics::removeBadFlowMasks().
    Flow error formula: mean((mask_flow - net_flow/5)^2) per instance,
    where net_flow/5 is the predicted flow scaled back to [-1,1].

    Args:
        labels: (H, W) int32 instance label map.
        raw_dy: (H, W) float32 raw model dy output (NOT /5 scaled).
        raw_dx: (H, W) float32 raw model dx output (NOT /5 scaled).
        flow_threshold: instances with mean error > this are removed.

    Returns:
        labels with bad-flow instances set to 0 (modified in place).
    """
    if flow_threshold <= 0 or labels.max() == 0:
        return labels

    net_dy = raw_dy.astype(np.float64) / 5.0
    net_dx = raw_dx.astype(np.float64) / 5.0
    mask_dy, mask_dx = masks_to_flows(labels)

    max_lbl = int(labels.max())
    error_sum = np.zeros(max_lbl + 1, dtype=np.float64)
    pixel_count = np.zeros(max_lbl + 1, dtype=np.int32)

    H, W = labels.shape
    for y in range(H):
        for x in range(W):
            lbl = labels[y, x]
            if lbl <= 0 or lbl > max_lbl:
                continue
            ddy = mask_dy[y, x] - net_dy[y, x]
            ddx = mask_dx[y, x] - net_dx[y, x]
            error_sum[lbl] += ddy * ddy + ddx * ddx
            pixel_count[lbl] += 1

    for lbl in range(1, max_lbl + 1):
        if pixel_count[lbl] <= 0:
            continue
        mean_err = error_sum[lbl] / pixel_count[lbl]
        if mean_err > flow_threshold:
            labels[labels == lbl] = 0

    return labels

import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, target_width=1000, target_height=1300):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    return warped

def auto_flatten_target_sheet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sheet_contour = None
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > (image.shape[0] * image.shape[1] * 0.2):
            sheet_contour = approx
            break
    if sheet_contour is not None:
        pts = sheet_contour.reshape(4, 2)
        return four_point_transform(image, pts)
    else:
        print("[INFO] Outer paper border not detected with high confidence. Using raw image.")
        return cv2.resize(image, (1000, 1300))

def draw_dotted_line(img, pt1, pt2, color=(0, 255, 255), gap=5):
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0:
        return
    num_dots = int(dist / gap)
    for i in range(num_dots + 1):
        r = i / max(1, num_dots)
        x = int((1 - r) * pt1[0] + r * pt2[0])
        y = int((1 - r) * pt1[1] + r * pt2[1])
        cv2.circle(img, (x, y), 2, color, -1)

def _nearest_neighbor_distance(idx, targets):
    tx, ty, _ = targets[idx]
    best = None
    for j, (ox, oy, _) in enumerate(targets):
        if j == idx:
            continue
        d = np.hypot(ox - tx, oy - ty)
        if best is None or d < best:
            best = d
    return best

def find_shot_hole(gray, tx, ty, r, search_radius, use_shift=True, debug_tag=None):
    h, w = gray.shape[:2]
    margin = int(search_radius) + 5
    y1, y2 = max(0, ty - margin), min(h, ty + margin)
    x1, x2 = max(0, tx - margin), min(w, tx + margin)
    roi = gray[y1:y2, x1:x2].astype(np.float32)

    rel_tx, rel_ty = tx - x1, ty - y1
    yy, xx = np.indices(roi.shape)
    dist_map = np.hypot(xx - rel_tx, yy - rel_ty)
    search_mask = (dist_map <= search_radius)

    if not search_mask.any():
        return None, None, (x1, y1)

    r_int = dist_map.astype(np.int32)
    max_r = int(search_radius) + 2
    overall_median = float(np.median(roi[search_mask]))
    expected = np.full(max_r + 1, overall_median, dtype=np.float32)
    
    for rad in range(max_r + 1):
        ring = (r_int == rad) & search_mask
        if ring.any():
            val = np.median(roi[ring])
            if rad < r * 0.85:
                val = min(val, overall_median + 15.0)
            expected[rad] = val
            
    deviation = np.abs(roi - expected[np.clip(r_int, 0, max_r)])
    if use_shift:
        devs = []
        for offset in [-2, -1, 0, 1, 2]:
            shifted_expected = expected[np.clip(r_int + offset, 0, max_r)]
            devs.append(np.abs(roi - shifted_expected))
        min_dev = np.minimum.reduce(devs)
        
        border_mask = (dist_map > r * 0.8) & (dist_map < r * 1.2)
        deviation = np.where(border_mask, min_dev, deviation)
        
    deviation[~search_mask] = 0

    vals = deviation[search_mask]
    if vals.size == 0 or vals.max() < 20:
        return None, None, (x1, y1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mm_per_px_est = 30.5 / (r * 2)
    expected_hole_area = np.pi * ((4.5 / 2) / mm_per_px_est) ** 2
    max_allowed_area = expected_hole_area * 6.0
    min_allowed_area = max(10.0, expected_hole_area * 0.12)

    def _extract_blobs(mask_u8):
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_allowed_area or area > max_allowed_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            if circularity > 0.20:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    dist_to_center = np.hypot(cx - rel_tx, cy - rel_ty)
                    found.append((cnt, area, circularity, dist_to_center))
        return found, mask_u8

    best_candidates, best_mask = [], None
    for pct in [98, 96, 93, 90, 87, 84]:
        thresh_val = max(20.0, float(np.percentile(vals, pct)))
        _, mask = cv2.threshold(deviation, thresh_val, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        candidates, closed_mask = _extract_blobs(mask)
        if candidates:
            best_candidates, best_mask = candidates, closed_mask
            break

    if debug_tag is not None:
        cv2.imwrite(f"debug/{debug_tag}_roi.png", roi.astype(np.uint8))
        cv2.imwrite(f"debug/{debug_tag}_response.png", cv2.normalize(deviation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        if best_mask is not None:
            cv2.imwrite(f"debug/{debug_tag}_mask.png", best_mask)

    if not best_candidates:
        return None, None, (x1, y1)

    best_candidates.sort(key=lambda t: t[1], reverse=True)
    top = best_candidates[:3]
    
    best_candidate = min(top, key=lambda t: t[3])
    largest_hole = best_candidate[0]

    M = cv2.moments(largest_hole)
    if M["m00"] != 0:
        hx = int(M["m10"] / M["m00"])
        hy = int(M["m01"] / M["m00"])
    else:
        hx, hy = rel_tx, rel_ty

    return hx, hy, (x1, y1)


BLANK_TARGET_PATH = "static/blank_target.png"
BLANK_TARGET_CENTER = (616, 612)   # measured center of the ring diagram, in px
# The diagram's solid black region ends at r=379.5px (measured). Scoring elsewhere
# in this file assumes the black bull is 30.5mm in diameter (mm_per_px = 30.5/(r*2)),
# i.e. a 15.25mm radius. Anchoring the diagram's black edge to that same 15.25mm
# keeps "inside/outside the black" on this overlay consistent with what's actually
# used to compute each shot's score.
BLANK_TARGET_PX_PER_MM = 379.5 / 15.25
PELLET_RADIUS_MM = 2.25            # 4.5mm pellet diameter, same assumption used in find_shot_hole

def _load_blank_target():
    img = cv2.imread(BLANK_TARGET_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not find blank target template at {BLANK_TARGET_PATH}")
    if img.shape[2] == 4:
        # flatten transparency onto white so it saves cleanly as a jpg
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        white_bg = np.full_like(bgr, 255)
        img = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    return img

def create_shot_overlay(shot_offsets_mm, output_path="scored_overlay.jpg"):
    """shot_offsets_mm: list of (dx_mm, dy_mm) per shot, relative to its own
    target's bull center. Plots them all onto the blank target template."""
    img = _load_blank_target()
    cx, cy = BLANK_TARGET_CENTER
    shot_radius_px = max(3, int(PELLET_RADIUS_MM * BLANK_TARGET_PX_PER_MM))

    for (dx_mm, dy_mm) in shot_offsets_mm:
        if dx_mm is None or dy_mm is None:
            continue

        dist_mm = np.hypot(dx_mm, dy_mm)
        # Real target scoring goes by the edge of the bullet hole nearest
        # center, not the hole's center point - pull the plotted point in by
        # the pellet radius so it lands in the same ring as the computed score.
        if dist_mm > PELLET_RADIUS_MM:
            scale = (dist_mm - PELLET_RADIUS_MM) / dist_mm
            dx_eff, dy_eff = dx_mm * scale, dy_mm * scale
        else:
            dx_eff, dy_eff = 0.0, 0.0

        px = int(round(cx + dx_eff * BLANK_TARGET_PX_PER_MM))
        py = int(round(cy + dy_eff * BLANK_TARGET_PX_PER_MM))

        cv2.circle(img, (px, py), shot_radius_px, (180, 120, 60), -1)
        cv2.circle(img, (px, py), shot_radius_px, (100, 60, 20), 2)

    cv2.imwrite(output_path, img)
    return img


def analyze_orion_target(image_path, output_path="scored_output_warped.jpg", overlay_output_path="scored_overlay.jpg", debug=False, debug_prefix=""):
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")

    img = auto_flatten_target_sheet(raw_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sighter_box = None
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bh) / bw if bw > 0 else 0
        area = bw * bh
        if (0.05 * h * w < area < 0.4 * h * w) and (1.2 < aspect_ratio < 2.5):
            if (0.25 * w < x + bw/2 < 0.75 * w) and (0.2 * h < y + bh/2 < 0.8 * h):
                sighter_box = (x, y, bw, bh)
                break

    _, dark_thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)

    bull_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_targets = []
    for cnt in bull_contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            (cx_float, cy_float), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx_float), int(cy_float)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            if circularity > 0.5:
                detected_targets.append((cx, cy, int(radius)))

    record_targets = []
    if sighter_box:
        bx, by, bw, bh = sighter_box
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
        for (tx, ty, r) in detected_targets:
            if not (bx <= tx <= bx + bw and by <= ty <= by + bh):
                record_targets.append((tx, ty, r))
    else:
        center_x, center_y = w / 2, h / 2
        sorted_by_dist = sorted(detected_targets, key=lambda t: (t[0]-center_x)**2 + (t[1]-center_y)**2)
        record_targets = sorted_by_dist[2:]

    record_targets = sorted(record_targets, key=lambda t: t[1])

    rows = []
    current_row = [record_targets[0]]
    for t in record_targets[1:]:
        if abs(t[1] - current_row[0][1]) < (h * 0.10):
            current_row.append(t)
        else:
            rows.append(current_row)
            current_row = [t]
    rows.append(current_row)

    ordered_targets = []
    for row in rows:
        sorted_row = sorted(row, key=lambda t: t[0])
        ordered_targets.extend(sorted_row)

    scores = []
    distances_mm = []
    shot_offsets_mm = []

    targets_to_score = ordered_targets[:10]

    for idx, (tx, ty, r) in enumerate(targets_to_score, start=1):
        target_center_global = (tx, ty)
        neighbor_dist = _nearest_neighbor_distance(idx - 1, targets_to_score)
        
        if neighbor_dist is not None:
            search_radius = max(r * 1.3, min(neighbor_dist * 0.48, r * 3.5))
        else:
            search_radius = r * 3.0

        tag = f"{debug_prefix}t{idx}" if debug else None
        
        hx, hy, offset_coords = find_shot_hole(gray, tx, ty, r, search_radius, use_shift=True, debug_tag=tag)

        dist_mm_prelim = 999.0
        if hx is not None:
            dist_px = np.sqrt((hx - (tx - offset_coords[0]))**2 + (hy - (ty - offset_coords[1]))**2)
            mm_per_px = 30.5 / (r * 2)
            dist_mm_prelim = dist_px * mm_per_px

        if hx is None or dist_mm_prelim > 27.0:
            border_search_radius = r * 1.3  
            tag_fallback = f"{tag}_fallback" if tag else None
            
            hx2, hy2, offset_coords2 = find_shot_hole(
                gray, tx, ty, r, 
                border_search_radius, 
                use_shift=False, 
                debug_tag=tag_fallback
            )
            
            if hx2 is not None:
                hx, hy, offset_coords = hx2, hy2, offset_coords2

        if hx is not None:
            x1, y1 = offset_coords
            shot_center_global = (x1 + hx, y1 + hy)
            dx_px = hx - (tx - x1)
            dy_px = hy - (ty - y1)
            dist_px = np.hypot(dx_px, dy_px)

            mm_per_px = 30.5 / (r * 2)
            dist_mm = dist_px * mm_per_px
            dx_mm = dx_px * mm_per_px
            dy_mm = dy_px * mm_per_px

            score = 10.9 - (dist_mm * 0.4)
            score = round(max(0.0, min(10.9, score)), 1)

            draw_dotted_line(img, target_center_global, shot_center_global, color=(0, 255, 255), gap=4)
            cv2.circle(img, target_center_global, 3, (255, 255, 0), -1)
            cv2.circle(img, shot_center_global, 4, (0, 0, 255), -1)

            label_text = f"#{idx}: {score:.1f} ({dist_mm:.2f}mm)"
        else:
            score = 0.0
            dist_mm = 0.0
            dx_mm = dy_mm = None
            label_text = f"#{idx}: {score:.1f}"

        scores.append(score)
        distances_mm.append(round(dist_mm, 2))
        shot_offsets_mm.append((dx_mm, dy_mm))

        label_pos = (tx - r - 10, ty - r - 8)
        # white outline first for contrast against any background, then navy fill on top
        cv2.putText(img, label_text, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, label_text, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), r, (255, 0, 0), 2)

    total_score = round(sum(scores), 1)
    cv2.imwrite(output_path, img)

    create_shot_overlay(shot_offsets_mm, overlay_output_path)

    return scores, distances_mm, total_score
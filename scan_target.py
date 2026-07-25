import cv2
import numpy as np

def order_points(pts):
    """
    Orders 4 coordinates in sequence: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts, target_width=1000, target_height=1300):
    """
    Warps and flattens a quadrilateral region in an image into a clean top-down view.
    """
    rect = order_points(pts)
    
    # Destination coordinates for uniform sheet resolution
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]], dtype="float32")

    # Perspective Transformation Matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    
    return warped

def auto_flatten_target_sheet(image):
    """
    Detects outer paper edges and un-skews/flattens the target sheet image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours representing paper sheet
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    sheet_contour = None
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        
        # Look for the largest 4-sided polygon
        if len(approx) == 4 and cv2.contourArea(cnt) > (image.shape[0] * image.shape[1] * 0.2):
            sheet_contour = approx
            break

    if sheet_contour is not None:
        pts = sheet_contour.reshape(4, 2)
        return four_point_transform(image, pts)
    else:
        # Fallback if paper borders aren't distinct (e.g. tight crop or dark background)
        print("[INFO] Outer paper border not detected with high confidence. Using raw image.")
        return cv2.resize(image, (1000, 1300))

def normalize_lighting(gray):
    """
    NEW (fixes scan-to-scan inconsistency): evens out shadows, glare, and
    vignetting across the photographed sheet before any thresholding happens.

    Every downstream detection step in this file used to compare pixels
    against fixed absolute brightness constants (100, 80, 170, 140...). Those
    only work for the exact lighting condition they were tuned on - a shadow
    from the phone or a hand, glare on one side, or angle-dependent falloff
    toward the edges of the frame will shift real pixel values enough to flip
    a detection, especially for targets far from the sheet's center.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) rescales
    contrast within small local tiles instead of across the whole image, so
    a shadow over the outer targets no longer makes that part of the sheet
    read differently than the rest. A light blur afterward keeps CLAHE from
    amplifying sensor noise in the flat white paper areas.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    equalized = clahe.apply(gray)
    return cv2.GaussianBlur(equalized, (3, 3), 0)

def draw_dotted_line(img, pt1, pt2, color=(0, 255, 255), gap=5):
    """Draws a dotted line between two point tuples (x, y)."""
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
    """Distance from targets[idx] to the closest other target center."""
    tx, ty, _ = targets[idx]
    best = None
    for j, (ox, oy, _) in enumerate(targets):
        if j == idx:
            continue
        d = np.hypot(ox - tx, oy - ty)
        if best is None or d < best:
            best = d
    return best

def _local_otsu(roi_gray, mask, fallback, lo, hi, inverse=False):
    """
    NEW: computes an Otsu threshold from only the pixels under `mask`,
    clamped to [lo, hi]. This replaces a single fixed brightness constant
    with one that adapts to each target's own local lighting.

    Falls back to `fallback` if there aren't enough pixels under the mask
    for a stable estimate (e.g. a target sitting right at the sheet edge
    with a clipped ROI). The [lo, hi] clamp keeps a near-uniform region
    (no real hole present) from letting Otsu drift to a wild, noise-picking
    value - it can adapt within a sane range, but not run away.
    """
    pixels = roi_gray[mask > 0]
    if pixels.size < 40:
        return fallback
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    cut, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255, flag | cv2.THRESH_OTSU)
    return int(np.clip(cut, lo, hi))

def find_shot_hole(gray, tx, ty, r, search_radius):
    """
    Locates the bullet hole near a target center.

    FIX (bug report: holes outside the black aiming mark were never detected):
    - `search_radius` now covers the full usable scoring area around the target
      (computed by the caller from the sheet's actual target spacing) instead of
      a fixed 0.92 * r, which previously discarded anything past the edge of the
      black circle before contours were even found.
    - Holes are now flagged as blobs that are either BRIGHTER or DARKER than
      their immediate local background, instead of only "brighter than 170".
      A hole punched through black ink exposes lighter backing (bright blob);
      a hole in the white paper area reads as a shadowed/torn dark blob. The
      old single-direction bright threshold only ever matched the first case.

    FIX (inconsistent scan-to-scan results, worse for outer targets):
    - The bright/dark cutoffs are no longer fixed constants (170 / 140).
      Each zone now runs `_local_otsu` on just the pixels in that target's own
      ROI, so the cutoff adapts to that target's local brightness instead of
      assuming the whole sheet is lit identically. The old constants remain
      as fallback/clamp bounds for safety.

    Returns (hx, hy) in ROI-relative coords, and the ROI offset (x1, y1), or
    (None, None), (None, None) if nothing found.
    """
    h, w = gray.shape[:2]
    margin = int(search_radius) + 5
    y1, y2 = max(0, ty - margin), min(h, ty + margin)
    x1, x2 = max(0, tx - margin), min(w, tx + margin)
    roi_gray = gray[y1:y2, x1:x2]

    rel_tx, rel_ty = tx - x1, ty - y1

    # Distance-from-center map, used to split the ROI into two zones so each
    # can be thresholded against its own known background instead of a blurred
    # estimate (a blur smears badly across the sharp printed black/white edge
    # and misreads that boundary itself as a "hole").
    yy, xx = np.indices(roi_gray.shape)
    dist_map = np.hypot(xx - rel_tx, yy - rel_ty)

    # Zone A: inside the printed black circle -> hole reads bright against ink.
    inside_mask = (dist_map <= r * 0.95).astype(np.uint8) * 255
    bright_cut = _local_otsu(roi_gray, inside_mask, fallback=170, lo=140, hi=220)
    _, bright_thresh = cv2.threshold(roi_gray, bright_cut, 255, cv2.THRESH_BINARY)
    bright_holes = cv2.bitwise_and(bright_thresh, inside_mask)

    # Zone B: outside the black circle, out to the usable search radius ->
    # hole reads as a shadowed/torn dark spot against the white paper.
    outside_mask = ((dist_map >= r * 1.05) & (dist_map <= search_radius)).astype(np.uint8) * 255
    dark_cut = _local_otsu(roi_gray, outside_mask, fallback=140, lo=100, hi=170, inverse=True)
    _, dark_thresh = cv2.threshold(roi_gray, dark_cut, 255, cv2.THRESH_BINARY_INV)
    dark_holes = cv2.bitwise_and(dark_thresh, outside_mask)

    hole_thresh = cv2.bitwise_or(bright_holes, dark_holes)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hole_thresh = cv2.morphologyEx(hole_thresh, cv2.MORPH_OPEN, kernel)

    hole_contours, _ = cv2.findContours(hole_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Circularity filter (same principle already used elsewhere in this file
    # for bullseye detection) rejects thin printed scoring-ring lines/numbers
    # in the white zone, which pass the dark threshold but aren't round blobs.
    
    valid_holes = []
    for cnt in hole_contours:
        area = cv2.contourArea(cnt)
        if area <= 15:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity > 0.35:
            valid_holes.append(cnt)

    if not valid_holes:
        return None, None, (x1, y1)

    largest_hole = max(valid_holes, key=cv2.contourArea)
    M = cv2.moments(largest_hole)
    if M["m00"] != 0:
        hx = int(M["m10"] / M["m00"])
        hy = int(M["m01"] / M["m00"])
    else:
        hx, hy = rel_tx, rel_ty

    return hx, hy, (x1, y1)

def analyze_orion_target(image_path, output_path="scored_output_warped.jpg", debug=False):
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")

    # 1. AUTO-PERSPECTIVE CORRECTION (Un-skew and flatten photo)
    img = auto_flatten_target_sheet(raw_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # NEW: normalize lighting before any detection happens. This is the main
    # fix for scan-to-scan inconsistency - see normalize_lighting() docstring.
    gray = normalize_lighting(gray)
    if debug:
        cv2.imwrite("debug_normalized.jpg", gray)

    h, w = gray.shape[:2]

    # 2. Locate central sighter box (SS targets)
    # CHANGED: fixed threshold of 100 -> Otsu, so the cutoff is picked per-photo
    # instead of assuming a fixed brightness level that may not hold for this
    # particular lighting condition.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    # 3. Detect target bullseyes & calculate centers
    # CHANGED: fixed threshold of 80 -> Otsu, same reasoning as above.
    _, dark_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        cv2.imwrite("debug_bullseye_thresh.jpg", dark_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)
    
    bull_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_targets = []
    for cnt in bull_contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                (cx, cy), _ = cv2.minEnclosingCircle(cnt)
                cx, cy = int(cx), int(cy)
                
            _, radius = cv2.minEnclosingCircle(cnt)
            circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0
            if circularity > 0.5:
                detected_targets.append((cx, cy, int(radius)))

    # 4. Exclude central sighter targets
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

    # 5. Sort 10 perimeter targets sequentially (1 through 10)
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

    # 6. Scoring & Deviation Distance Engine
    scores = []
    distances_mm = []

    targets_to_score = ordered_targets[:10]

    for idx, (tx, ty, r) in enumerate(targets_to_score, start=1):
        target_center_global = (tx, ty)

        # FIX: search radius now spans the real usable scoring area around this
        # target (roughly half the gap to its nearest neighbor on the sheet),
        # instead of a fixed 0.92 * r that clipped off anything outside the
        # black aiming mark. Falls back to a generous multiple of r if a
        # neighbor distance can't be determined (e.g. a lone target).
        neighbor_dist = _nearest_neighbor_distance(idx - 1, targets_to_score)
        if neighbor_dist is not None:
            search_radius = max(r * 1.3, min(neighbor_dist * 0.48, r * 3.5))
        else:
            search_radius = r * 3.0

        hx, hy, (x1, y1) = find_shot_hole(gray, tx, ty, r, search_radius)

        if hx is not None:
            shot_center_global = (x1 + hx, y1 + hy)
            dist_px = np.sqrt((hx - (tx - x1))**2 + (hy - (ty - y1))**2)
            
            # Scale calculation: 30.5mm = black aiming mark diameter
            mm_per_px = 30.5 / (r * 2)
            dist_mm = dist_px * mm_per_px
            
            # ISSF Decimal Score formula
            score = 10.9 - (dist_mm * 0.4)
            score = round(max(0.0, min(10.9, score)), 1)
            
            # Visual overlay
            draw_dotted_line(img, target_center_global, shot_center_global, color=(0, 255, 255), gap=4)
            cv2.circle(img, target_center_global, 3, (255, 255, 0), -1)
            cv2.circle(img, shot_center_global, 4, (0, 0, 255), -1)
            
            label_text = f"#{idx}: {score} ({dist_mm:.2f}mm)"
        else:
            score = 0.0
            dist_mm = 0.0
            label_text = f"#{idx}: {score}"
            
        scores.append(score)
        distances_mm.append(round(dist_mm, 2))
        
        cv2.putText(img, label_text, (tx - r - 10, ty - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(img, (tx, ty), r, (255, 0, 0), 2)

    total_score = round(sum(scores), 1)
    
    cv2.imwrite(output_path, img)

    # NEW: return the normalized/filtered grayscale image too (the same thing
    # debug=True saves to disk), so a caller (e.g. the web server) can show it
    # to the user alongside the scored overlay - this is the "filtered version"
    # of the photo, same idea as the test(2) example.
    return scores, distances_mm, total_score, gray

if __name__ == "__main__":
    individual_scores, distances, total, filtered_img = analyze_orion_target("image2.png", debug=True)
    
    print("\n--- TARGET SCORING RESULTS ---")
    for i in range(len(individual_scores)):
        print(f"Target {i+1}: Score = {individual_scores[i]} | Distance = {distances[i]} mm")
    print("------------------------------")
    print(f"TOTAL SCORE: {total} / 109.0\n")
import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage.exposure import rescale_intensity

def enhance_target_sheet(image):
    """
    CamScanner-style enhancement of the flattened target sheet.

    The goal (per the "Magic AI Pro" vibe) is a crisp high-contrast black &
    white scan: uneven lighting from a phone photo is flattened with a LOCAL
    Otsu threshold, so the printed black aiming marks / scoring rings and the
    shot holes read as near-pure black against a near-pure white paper
    background regardless of where the light fell on the sheet.

    Returns a uint8 single-channel (grayscale) image. Callers that need a BGR
    image for drawing (OpenCV overlay text/lines) should wrap the result with
    cv2.cvtColor(..., cv2.COLOR_GRAY2BGR).
    """
    # Work on a grayscale copy so the filter is purely intensity-based.
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Stretch the dynamic range first so the adaptive threshold operates on
    # the full 0-255 span (handy when a photo is washed out / low contrast).
    gray = rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)

    # Local Otsu: compute a threshold per block instead of one global cut.
    # Block size MUST be odd and >= the largest printed feature we want to
    # preserve as solid black (the aiming marks); 31px comfortably covers the
    # rings on a 1000x1300 flattened sheet while still adapting to shadows.
    block_size = 31
    local_thresh = threshold_local(gray, block_size, offset=10, method="gaussian")

    # threshold_local returns a per-pixel threshold array; pixels darker than
    # their local threshold become black (ink), the rest become white (paper).
    binary = (gray > local_thresh).astype(np.uint8) * 255

    # Light morphological cleanup: knock out single-pixel pepper/noise specks
    # without eroding the real shot holes (which are several px across).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

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

def find_shot_hole(gray, tx, ty, r, search_radius):
    """
    Locates the bullet hole near a target center.

    FIX (bug report: holes outside the black aiming mark were never detected):
    - `search_radius` now covers the full usable scoring area around the target
      (computed by the caller from the sheet's actual target spacing) instead of
      a fixed 0.92 * r, which previously discarded anything past the edge of the
      black circle before contours were even found.
    - Holes are now flagged as blobs that are either BRIGHTER or DARKER than
      their immediate local background (via a blurred local-background estimate),
      instead of only "brighter than 170". A hole punched through black ink
      exposes lighter backing (bright blob); a hole in the white paper area
      reads as a shadowed/torn dark blob. The old single-direction bright
      threshold only ever matched the first case.

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
    _, bright_thresh = cv2.threshold(roi_gray, 170, 255, cv2.THRESH_BINARY)
    bright_holes = cv2.bitwise_and(bright_thresh, inside_mask)

    # Zone B: outside the black circle, out to the usable search radius ->
    # hole reads as a shadowed/torn dark spot against the white paper.
    outside_mask = ((dist_map >= r * 1.05) & (dist_map <= search_radius)).astype(np.uint8) * 255
    _, dark_thresh = cv2.threshold(roi_gray, 140, 255, cv2.THRESH_BINARY_INV)
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

def analyze_orion_target(image_path, output_path="scored_output_warped.jpg"):
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")

    # 1. AUTO-PERSPECTIVE CORRECTION (Un-skew and flatten photo)
    img = auto_flatten_target_sheet(raw_img)

    # 1b. SCIKIT-IMAGE ENHANCEMENT — CamScanner-style high-contrast B&W.
    # The flattened sheet is run through a local Otsu threshold so the printed
    # marks and shot holes are near-pure black on white regardless of lighting,
    # which makes downstream bullseye / shot-hole detection more accurate. `gray`
    # (the filtered image) drives ALL detection below; `img` is rebuilt from it
    # as a BGR canvas so the final scored preview also reads as crisp B&W.
    gray = enhance_target_sheet(img)
    h, w = gray.shape[:2]
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. Locate central sighter box (SS targets)
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

    # 3. Detect target bullseyes & calculate centers
    _, dark_thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
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
    return scores, distances_mm, total_score

if __name__ == "__main__":
    individual_scores, distances, total = analyze_orion_target("image2.png")
    
    print("\n--- TARGET SCORING RESULTS ---")
    for i in range(len(individual_scores)):
        print(f"Target {i+1}: Score = {individual_scores[i]} | Distance = {distances[i]} mm")
    print("------------------------------")
    print(f"TOTAL SCORE: {total} / 109.0\n")    
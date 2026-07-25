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


def find_shot_hole(gray, tx, ty, r, search_radius, debug_tag=None):
    """
    Locates the bullet hole near a target center using radial-median
    background subtraction.

    Why this replaces the old "inside black circle -> bright test" /
    "outside on white paper -> dark test" zone split: that split left an
    unclassified ring of pixels right at the printed-circle boundary
    (roughly 0.95r-1.05r), and any hole straddling that boundary got
    half-judged by each test -- so neither half alone passed the
    area/circularity filter and the hole was missed entirely (scored 0.0).
    Even when a hole WAS detected, if only part of it fell within the
    "correct" zone, the centroid was computed from that partial blob only,
    dragging the reported center toward the edge of the true hole instead
    of its middle -- the "overshoots to the edge" symptom.

    A generic local-contrast method (e.g. top-hat/black-hat) was tried and
    rejected: because the target face is covered in real printed features
    (the outer black/white boundary, concentric dotted scoring rings), any
    method that flags "locally different from neighboring pixels" fires on
    those printed edges too, not just on holes -- it can't tell a printed
    ring from a punched hole.

    Radial-median subtraction fixes this by using the fact that the target
    face is radially symmetric: at any given radius from the bullseye
    center, the printed pattern (ink or paper, ring or gap) is the same at
    every angle. So for each radius we compute the *median* brightness over
    all angles at that radius -- this is robust to a single localized hole
    (a minority of angles) while representing the true printed background.
    Subtracting that expected-value-at-this-radius from the actual pixel
    isolates only things that break radial symmetry: the hole. This
    correctly ignores the boundary and the printed rings (which match their
    own radius's expected value almost everywhere) while still lighting up
    a hole wherever it sits, including squarely on the boundary itself.
    """
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
            expected[rad] = np.median(roi[ring])
    expected_map = expected[np.clip(r_int, 0, max_r)]

    deviation = np.abs(roi - expected_map)
    deviation[~search_mask] = 0

    vals = deviation[search_mask]
    if vals.size == 0 or vals.max() < 20:
        return None, None, (x1, y1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Expected real-hole size, from the known pellet caliber (4.5mm) and
    # this target's mm-per-pixel scale, used to sanity-check candidates so
    # we don't accept an implausibly huge or tiny blob (e.g. print smudges,
    # or a fleck of noise).
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
            if circularity > 0.25:
                found.append((cnt, area, circularity))
        return found, mask_u8

    # Adaptive percentile threshold: start strict (isolates only the
    # strongest anomaly) and relax progressively if nothing plausible is
    # found, so faint/partial holes are still recovered without letting
    # ordinary print noise through at the strict end.
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

    # Prefer the most circular candidate among the largest few -- a real
    # hole is round; leftover printed-number/ring fragments that slip
    # through the area filter tend to be more irregular.
    best_candidates.sort(key=lambda t: t[1], reverse=True)
    top = best_candidates[:3]
    largest_hole = max(top, key=lambda t: t[2])[0]

    M = cv2.moments(largest_hole)
    if M["m00"] != 0:
        hx = int(M["m10"] / M["m00"])
        hy = int(M["m01"] / M["m00"])
    else:
        hx, hy = rel_tx, rel_ty

    return hx, hy, (x1, y1)


def analyze_orion_target(image_path, output_path="scored_output_warped.jpg", debug=False, debug_prefix=""):
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

    targets_to_score = ordered_targets[:10]

    for idx, (tx, ty, r) in enumerate(targets_to_score, start=1):
        target_center_global = (tx, ty)

        neighbor_dist = _nearest_neighbor_distance(idx - 1, targets_to_score)
        if neighbor_dist is not None:
            search_radius = max(r * 1.3, min(neighbor_dist * 0.48, r * 3.5))
        else:
            search_radius = r * 3.0

        tag = f"{debug_prefix}t{idx}" if debug else None
        hx, hy, (x1, y1) = find_shot_hole(gray, tx, ty, r, search_radius, debug_tag=tag)

        if hx is not None:
            shot_center_global = (x1 + hx, y1 + hy)
            dist_px = np.sqrt((hx - (tx - x1))**2 + (hy - (ty - y1))**2)

            mm_per_px = 30.5 / (r * 2)
            dist_mm = dist_px * mm_per_px

            score = 10.9 - (dist_mm * 0.4)
            score = round(max(0.0, min(10.9, score)), 1)

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
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

def find_shot_hole(gray, tx, ty, r, search_radius):
    h, w = gray.shape[:2]
    margin = int(search_radius) + 5
    y1, y2 = max(0, ty - margin), min(h, ty + margin)
    x1, x2 = max(0, tx - margin), min(w, tx + margin)
    roi_gray = gray[y1:y2, x1:x2]

    rel_tx, rel_ty = tx - x1, ty - y1

    yy, xx = np.indices(roi_gray.shape)
    dist_map = np.hypot(xx - rel_tx, yy - rel_ty)

    # FIX (bug: holes on/near the black-circle border were missed or mis-centered):
    # The old code split the ROI into an inside zone (dist <= 0.95r, bright-hole
    # check) and an outside zone (dist >= 1.05r, dark-hole check), leaving the
    # annulus between 0.95r and 1.05r covered by NEITHER mask. Any hole that fell
    # in that gap - or straddled it - was invisible to at least one side, giving
    # a false 0.0 (hole entirely in the gap) or a lopsided centroid computed from
    # only the fragment that landed in one zone (hole straddling the gap).
    #
    # Fix: build an "ideal" reference image of this target - a black disk of
    # radius r on white, blurred to match the soft printed edge - and flag pixels
    # that deviate strongly from what that reference expects, over the WHOLE
    # search disk (no radius-based gap at all). Because the reference already
    # models the black/white transition at the true border, the border itself
    # produces ~0 deviation and stays quiet, while a real hole (bright breach in
    # the ink, or dark tear in the paper) deviates sharply no matter which side
    # of the printed edge it sits on.
    # Build the "expected" background directly from THIS photo's own pixels,
    # one thin radius ring at a time, using the median value around each ring.
    # A real hole only ever occupies a small arc of any given ring, so the
    # median stays locked onto the true background (black ink, white paper,
    # or the printed edge's transition band) no matter where that ring falls -
    # it needs no assumption about pure 0/255 levels (real photos rarely hit
    # those, given uneven lighting/exposure) and no guess at the edge's exact
    # blur width, so it can't drift out of sync with the true printed border
    # the way a synthetic reference circle can.
    r_bin = dist_map.astype(np.int32)
    max_bin = int(np.ceil(search_radius)) + 1
    radial_median = np.full(max_bin + 1, -1.0, dtype=np.float32)
    for rad in range(max_bin + 1):
        band_vals = roi_gray[r_bin == rad]
        if band_vals.size > 0:
            radial_median[rad] = np.median(band_vals)
    # fill any empty bins (can happen for tiny radii) from the nearest filled one
    for rad in range(1, max_bin + 1):
        if radial_median[rad] < 0:
            radial_median[rad] = radial_median[rad - 1]
    if radial_median[0] < 0:
        radial_median[0] = radial_median[radial_median >= 0][0] if np.any(radial_median >= 0) else 128.0

    ideal = radial_median[np.clip(r_bin, 0, max_bin)]
    deviation = roi_gray.astype(np.int16) - ideal.astype(np.int16)

    # Contrast-adaptive threshold, scaled off this target's own black/white
    # range so it still works under uneven lighting.
    black_level = float(np.median(roi_gray[dist_map < r * 0.7])) if np.any(dist_map < r * 0.7) else 20.0
    white_level = float(np.median(roi_gray[(dist_map > r * 1.15) & (dist_map <= search_radius)])) \
        if np.any((dist_map > r * 1.15) & (dist_map <= search_radius)) else 220.0
    dev_threshold = max(40.0, 0.4 * (white_level - black_level))

    search_mask = (dist_map <= search_radius).astype(np.uint8) * 255
    bright_holes = ((deviation > dev_threshold).astype(np.uint8)) * 255   # breach through black ink
    dark_holes = ((deviation < -dev_threshold).astype(np.uint8)) * 255    # tear/shadow in white paper

    hole_thresh = cv2.bitwise_or(bright_holes, dark_holes)
    hole_thresh = cv2.bitwise_and(hole_thresh, search_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hole_thresh = cv2.morphologyEx(hole_thresh, cv2.MORPH_OPEN, kernel)

    hole_contours, _ = cv2.findContours(hole_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        hx, hy, (x1, y1) = find_shot_hole(gray, tx, ty, r, search_radius)

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
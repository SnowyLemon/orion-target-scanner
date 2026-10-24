import cv2
import numpy as np

#the top left corner is the smallest x+y sum, and the bottom right corner is the largest x+y sum
#the top right corner is the smallest y-x difference, and the bottom left corner is the largest y-x difference
def order_points(pts):
    corner = np.zeros((4, 2), dtype = "float32")
    sum = pts.sum(axis=1)
    corner[0] = pts[np.argmin(sum)]
    corner[2] = pts[np.argmax(sum)]
    diff = np.diff(pts, axis=1)
    corner[1] = pts[np.argmin(diff)]
    corner[3] = pts[np.argmax(diff)]
    return corner

#It flattens a tilted image into a straight rectangle, this does make it SLIGHTLY less inaccurate,
#but it will make the scanning more easy to read as most images arent perfectly straight.
#Thus, overall it increases precision and accuracy of the scoring.
def four_point_transform(image, pts, target_width = 1000, target_height = 1300):
    corner = order_points(pts)
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corner, dst)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
    return warped

#converts to greyscale, blurs, finds edges, finds contours, and then finds the largest
#contour that is a rectangle and returns a flattened image of it.
def auto_flatten_target_sheet(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sheet_contour = None
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        #ensures only 4 cornered shapes that cover at least 20% of image are considered, or else
        #some unwanted stuff like stickers will be scanned and that messes it up
        if len(approx) == 4 and cv2.contourArea(cnt) > (image.shape[0] * image.shape[1] * 0.2):
            sheet_contour = approx
            break
    if sheet_contour is not None:
        pts = sheet_contour.reshape(4, 2)
        return four_point_transform(image, pts)
    else:
        print("[INFO] Outer paper border not detected with high confidence. Using raw image.")
        return cv2.resize(image, (1000, 1300))

#Draws the tiny dotted line between points to visualize the distance
def draw_dotted_line(img, pt1, pt2, color=(0, 255, 255), gap=5):
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0:
        return
    numdots = int(dist / gap)
    for i in range(numdots + 1):
        r = i / max(1, numdots )
        x = int((1 - r) * pt1[0] + r * pt2[0])
        y = int((1 - r) * pt1[1] + r * pt2[1])
        cv2.circle(img, (x, y), 2, color, -1)

#given a list of bullseyes, finds the distance to the nearest neighbor for a given index
def _nearest_neighbor_distance(idx, targets):
    tx, ty, _ = targets[idx]
    best = None
    for j, (ox, oy, _) in enumerate(targets):
        if j == idx:
            continue
        dist = np.hypot(ox - tx, oy - ty)
        if best is None or dist < best:
            best = dist
    return best

#given bullseye center and radius, tries to find the pellet hole nearest to it.
def find_shot_hole(grey, target_x, target_y, bull_radius, search_radius, use_shift=True, debug_tag=None):
    h, w = grey.shape[:2]
    crop_margin = int(search_radius) + 5 #the +5 is needed, else it often misscans
    y1, y2 = max(0, target_y - crop_margin), min(h, target_y + crop_margin)
    x1, x2 = max(0, target_x - crop_margin), min(w, target_x + crop_margin)
    crop = grey[y1:y2, x1:x2].astype(np.float32)
    bull_x, bull_y = target_x - x1, target_y - y1
    pixel_row, pixel_col = np.indices(crop.shape)
    dist_from_bullseye = np.hypot(pixel_col - bull_x, pixel_row - bull_y)
    search_zone = (dist_from_bullseye <= search_radius)
    if not search_zone.any():
        return None, None, (x1, y1)
    ring_number = dist_from_bullseye.astype(np.int32)
    max_ring = int(search_radius) + 2
    typical_brightness = float(np.median(crop[search_zone]))
    expected_brightness = np.full(max_ring + 1, typical_brightness, dtype=np.float32)
    for ring in range(max_ring + 1):
        ring_pixels = (ring_number == ring) & search_zone
        if ring_pixels.any():
            ring_brightness = np.median(crop[ring_pixels])
            if ring < bull_radius * 0.85:
                ring_brightness = min(ring_brightness, typical_brightness + 15.0)
            expected_brightness[ring] = ring_brightness
    deviation = np.abs(crop - expected_brightness[np.clip(ring_number, 0, max_ring)])
    if use_shift:
        differences_per_shift = []
        for ring_shift in [-2, -1, 0, 1, 2]:
            shifted_expected_brightness = expected_brightness[np.clip(ring_number + ring_shift, 0, max_ring)]
            differences_per_shift.append(np.abs(crop - shifted_expected_brightness))
        smallest_difference = np.minimum.reduce(differences_per_shift)

        border_mask = (dist_from_bullseye > bull_radius * 0.8) & (dist_from_bullseye < bull_radius * 1.2)
        deviation = np.where(border_mask, smallest_difference, deviation)
    deviation[~search_zone] = 0
    search_zone_differences = deviation[search_zone]
    if search_zone_differences.size == 0 or search_zone_differences.max() < 20:
        return None, None, (x1, y1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mm_per_pixel = 30.5 / (bull_radius * 2)
    expected_hole_area = np.pi * ((4.5 / 2) / mm_per_pixel) ** 2
    max_hole_area = expected_hole_area * 6.0
    min_hole_area = max(10.0, expected_hole_area * 0.12)
   
    def _find_hole_candidates(binary_mask):
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)
        outlines, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for outline in outlines:
            area = cv2.contourArea(outline)
            if area < min_hole_area or area > max_hole_area:
                continue
            perimeter = cv2.arcLength(outline, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            if circularity > 0.20:
                moments = cv2.moments(outline)
                if moments["m00"] != 0:
                    blob_x = moments["m10"] / moments["m00"]
                    blob_y = moments["m01"] / moments["m00"]
                    distance_to_bull = np.hypot(blob_x - bull_x, blob_y - bull_y)
                    candidates.append((outline, area, circularity, distance_to_bull))
        return candidates, binary_mask
    best_candidates, best_mask = [], None
    for percentile in [98, 96, 93, 90, 87, 84]:
        threshold_value = max(20.0, float(np.percentile(search_zone_differences, percentile)))
        _, difference_mask = cv2.threshold(deviation, threshold_value, 255, cv2.THRESH_BINARY)
        difference_mask = difference_mask.astype(np.uint8)
        candidates, cleaned_mask = _find_hole_candidates(difference_mask)
        if candidates:
            best_candidates, best_mask = candidates, cleaned_mask
            break
    if debug_tag is not None:
        cv2.imwrite(f"debug/{debug_tag}_roi.png", crop.astype(np.uint8))
        cv2.imwrite(f"debug/{debug_tag}_response.png", cv2.normalize(deviation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        if best_mask is not None:
            cv2.imwrite(f"debug/{debug_tag}_mask.png", best_mask)
    if not best_candidates:
        return None, None, (x1, y1)
    best_candidates.sort(key=lambda candidate: candidate[1], reverse=True)
    three_largest = best_candidates[:3]
    closest_candidate = min(three_largest, key=lambda candidate: candidate[3])
    hole_outline = closest_candidate[0]
    moments = cv2.moments(hole_outline)
    if moments["m00"] != 0:
        hole_x = int(moments["m10"] / moments["m00"])
        hole_y = int(moments["m01"] / moments["m00"])
    else:
        hole_x, hole_y = bull_x, bull_y
    return hole_x, hole_y, (x1, y1)

blank_target_path = "static/blank_target.png"
blank_target_center = (616, 612)  
blank_target_px_per_mm = 379.5 / 15.25
pellet_radius_mm = 2.25            

#loads the picture of empty target so it can be drawn later
def _load_blank_target():
    img = cv2.imread(blank_target_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not find blank target template at {blank_target_path}")
    if img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        opacity = img[:, :, 3:4].astype(np.float32) / 255.0
        white_bg = np.full_like(bgr, 255)
        img = (bgr * opacity + white_bg * (1 - opacity)).astype(np.uint8)
    return img

def create_shot_overlay(shot_offsets_mm, output_path="scored_overlay.jpg"):
    """shot_offsets_mm: list of (dx_mm, dy_mm) per shot, relative to its own
    target's bull center. Plots them all onto the blank target template."""
    img = _load_blank_target()
    center_x, center_y = blank_target_center
    shot_radius_px = max(3, int(pellet_radius_mm * blank_target_px_per_mm))
    for (dx_mm, dy_mm) in shot_offsets_mm:
        if dx_mm is None or dy_mm is None:
            continue
        dot_x = int(round(center_x + dx_mm * blank_target_px_per_mm))
        dot_y = int(round(center_y + dy_mm * blank_target_px_per_mm))
        cv2.circle(img, (dot_x, dot_y), shot_radius_px, (180, 120, 60), -1)
        cv2.circle(img, (dot_x, dot_y), shot_radius_px, (100, 60, 20), 2)
    cv2.imwrite(output_path, img)
    return img

#the server that server.py calls
def analyze_orion_target(image_path, output_path="scored_output_warped.jpg", overlay_output_path="scored_overlay.jpg", debug=False, debug_prefix=""):
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")
    img = auto_flatten_target_sheet(raw_img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape[:2]
    _, threshold = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY_INV)
    all_outlines, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sighter_box = None
    for outline in all_outlines:
        x, y, bw, bh = cv2.boundingRect(outline)
        height_to_width_ratio = float(bh) / bw if bw > 0 else 0
        box_area = bw * bh
        if (0.05 * h * w < box_area < 0.4 * h * w) and (1.2 < height_to_width_ratio < 2.5):
            if (0.25 * w < x + bw/2 < 0.75 * w) and (0.2 * h < y + bh/2 < 0.8 * h):
                sighter_box = (x, y, bw, bh)
                break
    _, dark_threshold = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY_INV)
    speck_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_threshold = cv2.morphologyEx(dark_threshold, cv2.MORPH_OPEN, speck_kernel)
    bull_outlines, _ = cv2.findContours(dark_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_targets = []
    for outline in bull_outlines:
        area = cv2.contourArea(outline)
        if area > 800:
            (center_x_float, center_y_float), radius = cv2.minEnclosingCircle(outline)
            bull_x, bull_y = int(center_x_float), int(center_y_float)
            perimeter = cv2.arcLength(outline, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            if circularity > 0.5:
                detected_targets.append((bull_x, bull_y, int(radius)))
    record_targets = []
    if sighter_box:
        sighter_x, sighter_y, sighter_width, sighter_height = sighter_box
        cv2.rectangle(img, (sighter_x, sighter_y), (sighter_x + sighter_width, sighter_y + sighter_height), (0, 165, 255), 2)
        for (bull_x, bull_y, bull_radius) in detected_targets:
            inside_sighter_box = (sighter_x <= bull_x <= sighter_x + sighter_width) and (sighter_y <= bull_y <= sighter_y + sighter_height)
            if not inside_sighter_box:
                record_targets.append((bull_x, bull_y, bull_radius))
    else:
        image_center_x, image_center_y = w / 2, h / 2
        bulls_by_distance_from_center = sorted(
            detected_targets,
            key=lambda bull: (bull[0] - image_center_x)**2 + (bull[1] - image_center_y)**2
        )
        record_targets = bulls_by_distance_from_center[2:]
    record_targets = sorted(record_targets, key=lambda bull: bull[1])
    rows = []
    current_row = [record_targets[0]]
    for bull in record_targets[1:]:
        if abs(bull[1] - current_row[0][1]) < (h * 0.10):
            current_row.append(bull)
        else:
            rows.append(current_row)
            current_row = [bull]
    rows.append(current_row)
    ordered_targets = []
    for row in rows:
        row_left_to_right = sorted(row, key=lambda bull: bull[0])
        ordered_targets.extend(row_left_to_right)
    scores = []
    distances_mm = []
    shot_offsets_mm = []
    bulls_to_score = ordered_targets[:10]
    for idx, (bull_x, bull_y, bull_radius) in enumerate(bulls_to_score, start=1):
        bull_center = (bull_x, bull_y)
        dist_to_bull = _nearest_neighbor_distance(idx - 1, bulls_to_score)
        if dist_to_bull is not None:
            search_radius = max(bull_radius * 1.3, min(dist_to_bull * 0.48, bull_radius * 3.5))
        else:
            search_radius = bull_radius * 3.0
        debug_name = f"{debug_prefix}t{idx}" if debug else None
        hole_x, hole_y, crop_corner = find_shot_hole(grey, bull_x, bull_y, bull_radius, search_radius, use_shift=True, debug_tag=debug_name)
        first_pass_distance_mm = 999.0
        if hole_x is not None:
            dist_px = np.sqrt((hole_x - (bull_x - crop_corner[0]))**2 + (hole_y - (bull_y - crop_corner[1]))**2)
            mm_per_pixel = 30.5 / (bull_radius * 2)
            first_pass_distance_mm = dist_px * mm_per_pixel
        if hole_x is None or first_pass_distance_mm > 27.0:
            tight_search_radius = bull_radius * 1.3
            fallback_debug_name = f"{debug_name}_fallback" if debug_name else None
            fallback_hole_x, fallback_hole_y, fallback_crop_corner = find_shot_hole(
                grey, bull_x, bull_y, bull_radius,
                tight_search_radius,
                use_shift=False,
                debug_tag=fallback_debug_name
            )
            if fallback_hole_x is not None:
                hole_x, hole_y, crop_corner = fallback_hole_x, fallback_hole_y, fallback_crop_corner
        if hole_x is not None:
            x1, y1 = crop_corner
            hole_center = (x1 + hole_x, y1 + hole_y)
            dx_px = hole_x - (bull_x - x1)
            dy_px = hole_y - (bull_y - y1)
            dist_px = np.hypot(dx_px, dy_px)
            mm_per_pixel = 30.5 / (bull_radius * 2)
            dist_mm = dist_px * mm_per_pixel
            dx_mm = dx_px * mm_per_pixel
            dy_mm = dy_px * mm_per_pixel
            score = 10.9 - (dist_mm * 0.4)
            score = round(max(0.0, min(10.9, score)), 1)
            draw_dotted_line(img, bull_center, hole_center, color=(0, 255, 255), gap=4)
            cv2.circle(img, bull_center, 3, (255, 255, 0), -1)
            cv2.circle(img, hole_center, 4, (0, 0, 255), -1)
            label_text = f"#{idx}: {score:.1f} ({dist_mm:.2f}mm)"
        else:
            score = 0.0
            dist_mm = 0.0
            dx_mm = dy_mm = None
            label_text = f"#{idx}: {score:.1f}"
        scores.append(score)
        distances_mm.append(round(dist_mm, 2))
        shot_offsets_mm.append((dx_mm, dy_mm))
        label_pos = (bull_x - bull_radius - 10, bull_y - bull_radius - 8)
        cv2.putText(img, label_text, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, label_text, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (bull_x, bull_y), bull_radius, (255, 0, 0), 2)
    total_score = round(sum(scores), 1)
    cv2.imwrite(output_path, img)
    create_shot_overlay(shot_offsets_mm, overlay_output_path)
    return scores, distances_mm, total_score
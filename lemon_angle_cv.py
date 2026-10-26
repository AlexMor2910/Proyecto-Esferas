import cv2
import numpy as np
import csv
from pathlib import Path

input_dir = Path('Unprocessed')
output_dir = Path('Processed')
output_dir.mkdir(parents=True, exist_ok=True)
min_lemon_area = 90
contact_tolerance_px = 15
persist_frames = 2


def masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 55, 35), (135, 255, 255))
    blue[:, int(w * .64):] = 0
    green = cv2.inRange(hsv, (28, 38, 30), (92, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return (cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k),
            cv2.morphologyEx(green, cv2.MORPH_OPEN, k))


def candidates(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


for VIDEO in input_dir.glob('*.mp4'):
    OUTPUT = output_dir / VIDEO.name
    CSV = output_dir / VIDEO.with_suffix('.csv').name

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {VIDEO}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    sphere_samples = []
    for _ in range(min(12, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ok, frame = cap.read()
        if not ok: break
        blue, _ = masks(frame)
        cc = candidates(blue, 1500)
        if cc:
            c = max(cc, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(c)
            if 35 < r < min(w, h)*.45:
                sphere_samples.append((x, y, r))
    if not sphere_samples:
        raise RuntimeError('Blue sphere not detected. Adjust HSV thresholds.')
    cx, cy, R = np.median(np.array(sphere_samples), axis=0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    records = []
    previous = None
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok, frame = cap.read()
        if not ok: break
        _, green = masks(frame)
        cc = candidates(green, min_lemon_area)
        options = []
        for c in cc:
            (x, y), r = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            if not (5 <= r <= 42): continue
            if y > h*.84: continue
            if np.hypot(x-cx, y-cy) < R*.75: continue
            options.append((c, x, y, r, area))
        if previous is not None:
            options = [o for o in options if np.hypot(o[1]-previous[0], o[2]-previous[1]) < 125]
        chosen = min(options, key=lambda o: np.hypot(o[1]-previous[0],o[2]-previous[1])) if (options and previous is not None) else (max(options, key=lambda o:o[4]) if options else None)
        canvas = np.zeros_like(frame)
        C = (int(round(cx)), int(round(cy)))
        cv2.circle(canvas, C, int(round(R)), (255, 130, 35), 2)
        cv2.circle(canvas, C, 4, (255, 130, 35), -1)
        angle = gap = lx = ly = radius = None
        if chosen:
            contour, lx, ly, radius, _ = chosen
            previous = (lx, ly)
            L = (round(lx), round(ly))
            cv2.drawContours(canvas, [contour], -1, (0, 220, 100), 2)
            cv2.circle(canvas, L, 3, (0, 255, 255), -1)
            cv2.line(canvas, C, L, (230, 230, 230), 1)
            cv2.line(canvas, C, (C[0], max(0, C[1]-int(R+radius+25))), (90, 90, 90), 1)
            angle = float(np.degrees(np.arctan2(lx-cx, cy-ly)))
            gap = float(np.hypot(lx-cx, ly-cy) - (R+radius))
            cv2.putText(canvas, f'Angle: {angle:.1f} deg', (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
            cv2.putText(canvas, f'Gap: {gap:+.1f} px', (15, 62), cv2.FONT_HERSHEY_SIMPLEX, .6, (210,210,210), 2)
        cv2.putText(canvas, f't = {i/fps:.3f} s', (15, h-18), cv2.FONT_HERSHEY_SIMPLEX, .6, (180,180,180), 2)
        records.append(dict(frame=i, time_s=i/fps, lemon_x=lx, lemon_y=ly, lemon_radius=radius, angle_deg=angle, gap_px=gap))
        writer.write(canvas)
    cap.release()
    writer.release()
    
    contact_seen = False
    separation_start = None
    streak = 0
    detach = None
    for row in records:
        g = row['gap_px']
        if g is None:
            streak = 0
            continue
        if abs(g) <= contact_tolerance_px:
            contact_seen = True
            streak = 0
        elif contact_seen and g > contact_tolerance_px:
            if streak == 0: separation_start = row
            streak += 1
            if streak >= persist_frames:
                detach = separation_start
                break
        else:
            streak = 0

    with CSV.open('w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=records[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(records)
    print(f'Sphere center=({cx:.1f},{cy:.1f}), radius={R:.1f} px')
    if detach:
        print(f'Estimated first separated frame: {detach["frame"]}, t={detach["time_s"]:.3f} s, angle={detach["angle_deg"]:.2f} degrees')
    else:
        print('No reliable detachment detected; inspect gap_px in CSV and tune thresholds.')
    print(f'Video: {OUTPUT}\nData: {CSV}')

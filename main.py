import streamlit as st
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# --- STREAMLIT CONFIG (Muss zwingend als erstes stehen) ---
st.set_page_config(page_title="F1 Track Optimizer", layout="centered")

# --- MATHEMATISCHE & PHYSIKALISCHE FUNKTIONEN (Exakt aus deinem Originalcode) ---
def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

def speed_to_color(speed, min_speed, max_speed):
    if max_speed <= min_speed:
        return "#00aaff"
    t = (speed - min_speed) / (max_speed - min_speed)
    t = max(0.0, min(1.0, t))
    stops = [
        (0.00, (20, 80, 255)),
        (0.25, (0, 200, 255)),
        (0.50, (0, 230, 80)),
        (0.75, (255, 230, 0)),
        (1.00, (255, 20, 20)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            local_t = (t - t0) / (t1 - t0)
            return lerp_color(c0, c1, local_t)
    return "#ff1414"

def generate_spline_line(centers, offsets, angles, track_w, resolution=600):
    half_w = track_w / 2.0
    kp = []
    for i in range(len(centers)):
        n = np.array([np.cos(angles[i]), np.sin(angles[i])])
        kp.append(centers[i] + n * (offsets[i] * half_w))
    kp = np.array(kp)
    kp_closed = np.vstack([kp, kp[0]])
    t = np.arange(len(kp_closed))
    cs_x = CubicSpline(t, kp_closed[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t, kp_closed[:, 1], bc_type='periodic')
    t_new = np.linspace(0, len(kp), resolution)
    return np.column_stack([cs_x(t_new), cs_y(t_new)])

def calculate_lap_telemetry(smooth_line):
    n = len(smooth_line)
    if n < 3: 
        return [100.0] * n, [0.0] * n
    
    MU, G, MAX_ACCEL, MAX_BRAKE, MAX_SPEED, MIN_SPEED, PIXEL_SCALE = 3.5, 9.81, 320.0, 480.0, 520.0, 55.0, 0.35
    dists = np.zeros(n)
    for i in range(1, n):
        dists[i] = dists[i - 1] + np.linalg.norm(smooth_line[i] - smooth_line[i - 1])
        
    radii = np.full(n, 9999.0)
    for i in range(n):
        p1, p2, p3 = smooth_line[i - 1], smooth_line[i], smooth_line[(i + 1) % n]
        a, b = p2 - p1, p3 - p1
        cross = abs(a[0] * b[1] - a[1] * b[0])
        if cross > 0.01:
            r = (np.linalg.norm(p1 - p2) * np.linalg.norm(p2 - p3) * np.linalg.norm(p3 - p1)) / (2.0 * cross + 1e-9)
            radii[i] = min(r / PIXEL_SCALE, 1500.0)
            
    v_max_corner = np.sqrt(MU * G * radii) / PIXEL_SCALE
    v_max_corner = np.clip(v_max_corner, MIN_SPEED, MAX_SPEED)
    
    speeds = v_max_corner.copy()
    for i in range(n):
        ni = (i + 1) % n
        seg_dist = np.linalg.norm(smooth_line[ni] - smooth_line[i])
        if seg_dist < 0.001: continue
        speeds[ni] = min(speeds[ni], np.sqrt(speeds[i] ** 2 + 2.0 * MAX_ACCEL * seg_dist), MAX_SPEED)
        
    for _ in range(3):
        for i in range(n - 1, -1, -1):
            ni = (i + 1) % n
            seg_dist = np.linalg.norm(smooth_line[ni] - smooth_line[i])
            if seg_dist < 0.001: continue
            speeds[i] = min(speeds[i], np.sqrt(speeds[ni] ** 2 + 2.0 * MAX_BRAKE * seg_dist), MAX_SPEED)
            
    return speeds.tolist(), dists.tolist()

def calculate_lap_time(smooth_line, offsets):
    if np.any(np.abs(offsets) > 0.97): return float('inf')
    speeds, _ = calculate_lap_telemetry(smooth_line)
    total_time = 0.0
    n = len(smooth_line)
    for i in range(n):
        seg = np.linalg.norm(smooth_line[(i + 1) % n] - smooth_line[i])
        total_time += seg / max(speeds[i], 1.0)
    return total_time

def recalculate_auto_angles(points):
    n = len(points)
    angles = [0.0] * n
    if n < 2: return angles
    pts = np.array(points, dtype=float)
    if n >= 3:
        pts_c = np.vstack([pts[-1], pts, pts[0]])
        t = np.arange(len(pts_c))
        try:
            cs_x = CubicSpline(t, pts_c[:, 0])
            cs_y = CubicSpline(t, pts_c[:, 1])
            for i in range(n):
                angles[i] = np.arctan2(cs_y(i + 1, 1), cs_x(i + 1, 1)) + np.pi / 2.0
        except:
            for i in range(n):
                ni, pi_ = (i + 1) % n, (i - 1) % n
                diff = pts[ni] - pts[pi_]
                angles[i] = np.arctan2(diff[1], diff[0]) + np.pi / 2.0
    else:
        for i in range(n):
            ni, pi_ = (i + 1) % n, (i - 1) % n
            diff = pts[ni] - pts[pi_]
            angles[i] = np.arctan2(diff[1], diff[0]) + np.pi / 2.0
    return angles

# --- STREAMLIT OBERFLÄCHE (SESSION STATE) ---
st.title("🏎️ F1 Track Optimizer (Web)")
st.subheader("Racing Line + Speed Heatmap auf dem Smartphone")

# Initialisierung der Track-Punkte im Speicher
if 'track_points' not in st.session_state:
    # Eine Beispiel-Rennstrecke als Standard-Vorgabe
    st.session_state.track_points = [[200, 150], [400, 100], [600, 180], [700, 400], [500, 500], [250, 420]]
if 'optimized' not in st.session_state:
    st.session_state.optimized = False

# Sidebar Steuerelemente
st.sidebar.header("🔧 Einstellungen")
track_w = st.sidebar.slider("Streckenbreite", 15, 120, 50)
generations_input = st.sidebar.slider("Anzahl Generationen (Qualität)", 10, 200, 50)

st.sidebar.write("### 📍 Streckenpunkte bearbeiten")
# Auf Handys sind Touch-Klicks ungenau, daher hier als Textfeld editierbar!
points_text = st.sidebar.text_area("Punkte als X,Y (Ein Punkt pro Zeile)", 
                                   value="\n".join([f"{p[0]},{p[1]}" for p in st.session_state.track_points]),
                                   height=150)

# Eingabe-Text in Koordinaten-Arrays umwandeln
try:
    new_pts = []
    for line in points_text.strip().split("\n"):
        if "," in line:
            x, y = map(int, line.split(","))
            new_pts.append([x, y])
    if len(new_pts) >= 3:
        st.session_state.track_points = new_pts
except:
    st.sidebar.error("Fehler im Koordinaten-Format!")

# Optimierungs-Triggern
if st.sidebar.button("▶ IDEALLINIE BERECHNEN", type="primary"):
    with st.spinner("Genetischer Algorithmus rechnet..."):
        centers = np.array(st.session_state.track_points, dtype=float)
        n_pts = len(centers)
        angles = recalculate_auto_angles(st.session_state.track_points)
        
        best_offsets = np.zeros(n_pts)
        best_line = generate_spline_line(centers, best_offsets, angles, track_w)
        best_time = calculate_lap_time(best_line, best_offsets)
        
        # Genetische Evolution (angepasst für schnelle Web-Ladezeiten)
        for gen in range(generations_input):
            mutation_strength = max(0.02, 0.4 * (0.985 ** gen))
            for _ in range(25): 
                test = best_offsets.copy()
                for k in range(n_pts):
                    if np.random.rand() < 0.4:
                        test[k] += np.random.normal(0, mutation_strength)
                test = np.clip(test, -0.95, 0.95)
                line = generate_spline_line(centers, test, angles, track_w)
                t = calculate_lap_time(line, test)
                
                if t < best_time:
                    best_time = t
                    best_offsets = test
                    best_line = line
                    
        st.session_state.best_line = best_line
        st.session_state.best_time = best_time
        st.session_state.optimized = True
        st.sidebar.success(f"Optimiert! Zeit: {best_time:.3f}s")

# --- KANVAS ZEICHNEN (MATPLOTLIB) ---
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')

# Strecken-Mittelspur
pts = np.array(st.session_state.track_points)
pts_closed = np.vstack([pts, pts[0]])
ax.plot(pts_closed[:,0], pts_closed[:,1], color='#444444', linestyle='--', label='Centerline')
ax.scatter(pts[:,0], pts[:,1], color='#0078d7', s=120, zorder=5)

# Wenn optimiert, Heatmap-Linie zeichnen
if st.session_state.optimized:
    line_pts = st.session_state.best_line
    speeds, _ = calculate_lap_telemetry(line_pts)
    s_min, s_max = min(speeds), max(speeds)
    
    for i in range(len(line_pts) - 1):
        c = speed_to_color(speeds[i], s_min, s_max)
        ax.plot(line_pts[i:i+2, 0], line_pts[i:i+2, 1], color=c, linewidth=4)
    # Letztes Segment schließen
    c = speed_to_color(speeds[-1], s_min, s_max)
    ax.plot([line_pts[-1, 0], line_pts[0, 0]], [line_pts[-1, 1], line_pts[0, 1]], color=c, linewidth=4)

ax.invert_yaxis() # Tkinter Nullpunkt oben links simulieren
ax.axis('off')
st.pyplot(fig)

if st.session_state.optimized:
    st.metric(label="🏎️ Berechnete Rundenzeit", value=f"{st.session_state.best_time:.3f} s")
else:
    st.info("Punkte in der linken Sidebar anpassen und auf Berechnen drücken!")

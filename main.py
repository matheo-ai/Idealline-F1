import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from PIL import Image
import os

# ─────────────────────────────────────────────────────────────────────────
# MATHE & PHYSIK LOGIK (Originale Logik exakt beibehalten)
# ─────────────────────────────────────────────────────────────────────────

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgb({r},{g},{b})"

def speed_to_color(speed, min_speed, max_speed):
    if max_speed <= min_speed:
        return "rgb(0,170,255)"
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
    return "rgb(255,20,20)"

def recalculate_auto_angles(points):
    n = len(points)
    if n < 2: return [0.0] * n
    pts = np.array(points, dtype=float)
    angles = [0.0] * n
    
    if n >= 3:
        pts_c = np.vstack([pts[-1], pts, pts[0]])
        t = np.arange(len(pts_c))
        try:
            cs_x = CubicSpline(t, pts_c[:, 0])
            cs_y = CubicSpline(t, pts_c[:, 1])
            for i in range(n):
                dx = cs_x(i + 1, 1)
                dy = cs_y(i + 1, 1)
                angles[i] = np.arctan2(dy, dx) + np.pi / 2.0
        except:
            for i in range(n):
                ni = (i + 1) % n
                pi_ = (i - 1) % n
                diff = pts[ni] - pts[pi_]
                angles[i] = np.arctan2(diff[1], diff[0]) + np.pi / 2.0
    else:
        for i in range(n):
            ni = (i + 1) % n
            diff = pts[ni] - pts[i]
            angles[i] = np.arctan2(diff[1], diff[0]) + np.pi / 2.0
    return angles

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
    if n < 3: return [100.0] * n, [0.0] * n
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

def calculate_lap_time(smooth_line, offsets, track_w):
    if np.any(np.abs(offsets) > 0.97): return float('inf')
    speeds, _ = calculate_lap_telemetry(smooth_line)
    total_time = 0.0
    n = len(smooth_line)
    for i in range(n):
        seg = np.linalg.norm(smooth_line[(i + 1) % n] - smooth_line[i])
        total_time += seg / max(speeds[i], 1.0)
    return total_time

# ─────────────────────────────────────────────────────────────────────────
# STREAMLIT INTERFACE (Ersatzt für Tkinter)
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="F1 Track Optimizer")
st.title("🏁 F1 Track Optimizer – Racing Line")

# Bild laden
skript_ordner = os.path.dirname(os.path.abspath(__file__))
vollstaendiger_pfad = os.path.join(skript_ordner, "IMG_3654.png")

try:
    bg_image = Image.open(vollstaendiger_pfad)
    img_w, img_h = bg_image.size
except FileNotFoundError:
    img_w, img_h = 1000, 700
    bg_image = None

# Festgelegte Standardpunkte aus deinem Setup (Falls leer gestartet wird)
default_points = [[200, 550], [350, 200], [700, 250], [800, 600], [500, 650]]

if 'track_points' not in st.session_state:
    st.session_state.track_points = default_points
    st.session_state.manual_angles = [False] * len(default_points)
    st.session_state.marker_angles = recalculate_auto_angles(default_points)
    st.session_state.best_line = None

# Sidebar Controls
st.sidebar.header("Optionen")
track_w = st.sidebar.slider("Streckenbreite", 15, 120, 50)
steps = st.sidebar.number_input("Optimierungs-Schritte", 10, 200, 50)

# Berechnungs-Trigger
if st.sidebar.button("▶ IDEALLINIE BERECHNEN"):
    with st.spinner('Berechne optimale Rennlinie...'):
        centers = np.array(st.session_state.track_points, dtype=float)
        num_pts = len(centers)
        best_offsets = np.zeros(num_pts)
        angles = st.session_state.marker_angles
        
        best_line_points = generate_spline_line(centers, best_offsets, angles, track_w)
        best_time = calculate_lap_time(best_line_points, best_offsets, track_w)
        
        # Vereinfachte genetische Schleife für die Cloud-Ausführung
        for gen in range(int(steps)):
            mutation_strength = max(0.02, 0.4 * (0.985 ** gen))
            for _ in range(30):
                test = best_offsets.copy()
                for k in range(num_pts):
                    if np.random.rand() < 0.4:
                        test[k] += np.random.normal(0, mutation_strength)
                test = np.clip(test, -0.95, 0.95)
                line = generate_spline_line(centers, test, angles, track_w)
                t = calculate_lap_time(line, test, track_w)
                
                if t < best_time:
                    best_time = t
                    best_offsets = test
                    best_line_points = line
                    
        st.session_state.best_line = best_line_points
        st.sidebar.success(f"Optimiert! Zeit: {best_time:.3f}s")

# --- PLOTTY GRAPH ERSTELLEN ---
fig = go.Figure()

if bg_image:
    fig.add_layout_image(
        dict(
            source=bg_image, xref="x", yref="y", x=0, y=img_h,
            sizex=img_w, sizey=img_h, sizing="stretch", opacity=1, layer="below"
        )
    )
else:
    fig.update_layout(plot_bgcolor='#111111')

# Zeichne Ideallinie (Heatmap)
if st.session_state.best_line is not None:
    line_pts = st.session_state.best_line
    speeds, _ = calculate_lap_telemetry(line_pts)
    s_min, s_max = min(speeds), max(speeds)
    for i in range(len(line_pts) - 1):
        color = speed_to_color(speeds[i], s_min, s_max)
        fig.add_trace(go.Scatter(
            x=[line_pts[i, 0], line_pts[i+1, 0]], y=[line_pts[i, 1], line_pts[i+1, 1]],
            mode='lines', line=dict(color=color, width=4), hoverinfo='skip', showlegend=False
        ))

# Zeichne Strecken-Marker
half_w = track_w / 2.0
for i, p in enumerate(st.session_state.track_points):
    ang = st.session_state.marker_angles[i]
    n = np.array([np.cos(ang), np.sin(ang)])
    pl = p + n * half_w
    pr = p - n * half_w
    fig.add_trace(go.Scatter(x=[pl[0], pr[0]], y=[pl[1], pr[1]], mode='lines', line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode='markers+text', text=[str(i+1)], textposition="top center", marker=dict(color='#0078d7', size=8), showlegend=False))

fig.update_xaxes(range=[0, img_w], showgrid=False, zeroline=False, visible=False)
fig.update_yaxes(range=[img_h, 0], showgrid=False, zeroline=False, visible=False)
fig.update_layout(width=img_w, height=img_h, margin=dict(l=0, r=0, t=0, b=0))

st.plotly_chart(fig, use_container_width=True)

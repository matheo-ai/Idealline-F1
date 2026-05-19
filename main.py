import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os
import time
from scipy.interpolate import CubicSpline


def lerp_color(c1, c2, t):
    """Interpoliert zwischen zwei RGB-Farben."""
    t = max(0.0, min(1.0, t))
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def speed_to_color(speed, min_speed, max_speed):
    """Farbramp: Langsam = Blau -> Gelb -> Grün -> Rot = Schnell."""
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


class TrackOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("F1 Track Optimizer – Racing Line + Speed Heatmap")

        # Bild initialisieren
        skript_ordner = os.path.dirname(os.path.abspath(__file__))
        dateiname = "IMG_3654.png"
        vollstaendiger_pfad = os.path.join(skript_ordner, dateiname)

        try:
            self.bg_image = Image.open(vollstaendiger_pfad)
            self.img_w, self.img_h = self.bg_image.size
        except FileNotFoundError:
            self.img_w, self.img_h = 1000, 700
            self.bg_image = Image.new("RGB", (self.img_w, self.img_h), (20, 20, 20))

        # Core-Variablen
        self.track_points = []
        self.marker_angles = []
        self.manual_angles = []  # Trackt, welche Marker manuell gedreht wurden (True/False)

        self.selected_point = None
        self.edit_submode = None  # "move" oder "rotate"
        self.is_optimizing = False
        self.show_ui_elements = True

        # Optimierungs-State
        self.centers = None
        self.best_offsets = None
        self.best_line_points = None
        self.best_time = float('inf')
        self.generation = 0

        self.line_segment_ids = []

        # Auto
        self.car_id_body = None
        self.car_id_light = None
        self.car_distance = 0.0
        self.car_speeds = []
        self.cum_distances = []
        self.last_frame_time = time.time()

        # ── Sidebar ──────────────────────────────────────────────────────────
        self.sidebar = ttk.Frame(root, padding="15")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(self.sidebar, text="F1 TRACK OPTIMIZER", font=("Arial", 13, "bold")).pack(pady=(0, 12))

        ttk.Label(self.sidebar, text="STRECKE:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(4, 2))
        ttk.Button(self.sidebar, text="🖼️ Strecke hochladen", command=self.upload_track_image).pack(fill=tk.X,
                                                                                                    pady=(0, 10))

        ttk.Label(self.sidebar, text="WERKZEUGE:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.editor_mode = tk.StringVar(value="place")
        ttk.Radiobutton(self.sidebar, text="➕ Marker setzen", variable=self.editor_mode,
                        value="place", command=self.clear_selection).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.sidebar, text="✏️ Marker ziehen/drehen", variable=self.editor_mode,
                        value="edit").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.sidebar, text="❌ Marker löschen", variable=self.editor_mode,
                        value="delete", command=self.clear_selection).pack(anchor=tk.W, pady=2)

        ttk.Separator(self.sidebar).pack(fill=tk.X, pady=10)

        ttk.Label(self.sidebar, text="Streckenbreite:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.width_slider = ttk.Scale(self.sidebar, from_=15, to=120, value=50,
                                      command=self.update_canvas_drawings)
        self.width_slider.pack(fill=tk.X, pady=4)

        ttk.Separator(self.sidebar).pack(fill=tk.X, pady=10)

        self.play_btn = ttk.Button(self.sidebar, text="▶ IDEALLINIE BERECHNEN",
                                   command=self.toggle_optimization)
        self.play_btn.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(self.sidebar, text="🔄 Linie zurücksetzen",
                   command=self.reset_only_line).pack(fill=tk.X, pady=3)
        ttk.Button(self.sidebar, text="👁 Marker ein/ausblenden",
                   command=self.toggle_ui_visibility).pack(fill=tk.X, pady=3)
        ttk.Button(self.sidebar, text="🗑 Alles zurücksetzen",
                   command=self.reset_track).pack(fill=tk.X, pady=3)

        ttk.Separator(self.sidebar).pack(fill=tk.X, pady=10)

        # Legende
        ttk.Label(self.sidebar, text="SPEED-HEATMAP:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        legend_canvas = tk.Canvas(self.sidebar, width=160, height=20, highlightthickness=0)
        legend_canvas.pack(pady=4)
        for i in range(160):
            t = i / 159
            c = speed_to_color(t, 0.0, 1.0)
            legend_canvas.create_line(i, 0, i, 20, fill=c, width=1)

        leg_frame = ttk.Frame(self.sidebar)
        leg_frame.pack(fill=tk.X)
        ttk.Label(leg_frame, text="Langsam", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Label(leg_frame, text="Schnell", font=("Arial", 8)).pack(side=tk.RIGHT)

        ttk.Separator(self.sidebar).pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(self.sidebar, text="Bereit – Marker setzen",
                                      font=("Arial", 10, "bold"), foreground="#888888",
                                      wraplength=170, justify=tk.LEFT)
        self.status_label.pack(pady=5)

        self.gen_label = ttk.Label(self.sidebar, text="", font=("Arial", 9), foreground="#555555")
        self.gen_label.pack()

        # ── Canvas ───────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(root, width=self.img_w, height=self.img_h,
                                bg="#111111", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.tk_img = ImageTk.PhotoImage(self.bg_image)
        self.bg_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img, tags="background")

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        self.animate_car()

    # ─────────────────────────────────────────────────────────────────────────
    # BILD HOCHLADEN
    # ─────────────────────────────────────────────────────────────────────────
    def upload_track_image(self):
        dateipfad = filedialog.askopenfilename(
            title="Wähle ein Streckenbild aus",
            filetypes=[("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if dateipfad:
            try:
                neu_bild = Image.open(dateipfad)
                self.bg_image = neu_bild
                self.img_w, self.img_h = self.bg_image.size

                self.canvas.config(width=self.img_w, height=self.img_h)
                self.tk_img = ImageTk.PhotoImage(self.bg_image)
                self.canvas.itemconfig(self.bg_image_id, image=self.tk_img)

                self.reset_track()
                self.status_label.config(text="Neues Bild geladen!", foreground="#00ff88")
            except Exception as e:
                self.status_label.config(text=f"Fehler beim Laden: {e}", foreground="#ff4444")

    # ─────────────────────────────────────────────────────────────────────────
    # WINKEL-BERECHNUNG (Modifiziert: Ignoriert manuell veränderte Marker)
    # ─────────────────────────────────────────────────────────────────────────
    def recalculate_auto_angles(self):
        n = len(self.track_points)
        if n == 0: return
        if n < 2: return

        # Fallback-Winkel generieren falls ungenügend Punkte da sind
        pts = np.array(self.track_points, dtype=float)
        backup_angles = [0.0] * n

        if n >= 3:
            pts_c = np.vstack([pts[-1], pts, pts[0]])
            t = np.arange(len(pts_c))
            try:
                cs_x = CubicSpline(t, pts_c[:, 0])
                cs_y = CubicSpline(t, pts_c[:, 1])
                for i in range(n):
                    dx = cs_x(i + 1, 1)
                    dy = cs_y(i + 1, 1)
                    backup_angles[i] = np.arctan2(dy, dx) + np.pi / 2.0
            except Exception:
                for i in range(n):
                    ni = (i + 1) % n
                    pi_ = (i - 1) % n
                    diff = pts[ni] - pts[pi_]
                    backup_angles[i] = np.arctan2(diff[1], diff[0]) + np.pi / 2.0
        else:
            for i in range(n):
                ni = (i + 1) % n
                pi_ = (i - 1) % n
                diff = pts[ni] - pts[pi_]
                backup_angles[i] = np.arctan2(diff[1], diff[0]) + np.pi / 2.0

        # Nur Winkel überschreiben, die NICHT händisch gedreht wurden
        for i in range(n):
            if not self.manual_angles[i]:
                self.marker_angles[i] = backup_angles[i]

    # ─────────────────────────────────────────────────────────────────────────
    # CANVAS-INTERAKTION
    # ─────────────────────────────────────────────────────────────────────────
    def clear_selection(self):
        self.selected_point = None
        self.edit_submode = None
        self.update_canvas_drawings()

    def on_canvas_click(self, event):
        if self.is_optimizing or not self.show_ui_elements: return
        mode = self.editor_mode.get()
        pos = np.array([event.x, event.y])
        track_w = self.width_slider.get()
        half_w = track_w / 2.0

        clicked_idx = None
        submode = None

        for idx, pt in enumerate(self.track_points):
            p = np.array(pt)
            if np.linalg.norm(p - pos) < 18:
                clicked_idx = idx
                submode = "move"
                break

            angle = self.marker_angles[idx]
            n = np.array([np.cos(angle), np.sin(angle)])
            pl = p + n * half_w
            pr = p - n * half_w

            if np.linalg.norm(pl - pos) < 15 or np.linalg.norm(pr - pos) < 15:
                clicked_idx = idx
                submode = "rotate"
                break

        if mode == "edit":
            if clicked_idx is not None:
                self.selected_point = clicked_idx
                self.edit_submode = submode
                if self.edit_submode == "move":
                    self.track_points[self.selected_point] = [event.x, event.y]
                    self.recalculate_auto_angles()
                elif self.edit_submode == "rotate":
                    self.manual_angles[self.selected_point] = True  # Sperre für Auto-Berechnung aktivieren!
                    p = np.array(self.track_points[self.selected_point])
                    diff = pos - p
                    self.marker_angles[self.selected_point] = np.arctan2(diff[1], diff[0])
            else:
                self.selected_point = None
                self.edit_submode = None
            self.update_canvas_drawings()

        elif mode == "place":
            if clicked_idx is None:
                self.track_points.append([event.x, event.y])
                self.marker_angles.append(0.0)
                self.manual_angles.append(False)  # Neuer Marker startet im Auto-Modus
                self.recalculate_auto_angles()
                self.update_canvas_drawings()

        elif mode == "delete":
            if clicked_idx is not None:
                self.track_points.pop(clicked_idx)
                self.marker_angles.pop(clicked_idx)
                self.manual_angles.pop(clicked_idx)
                self.selected_point = None
                self.edit_submode = None
                self.recalculate_auto_angles()
                self.update_canvas_drawings()

    def on_mouse_move(self, event):
        if self.is_optimizing or not self.show_ui_elements: return
        if self.editor_mode.get() == "edit" and self.selected_point is not None:
            x = max(0, min(event.x, self.img_w))
            y = max(0, min(event.y, self.img_h))
            pos = np.array([x, y])
            p = np.array(self.track_points[self.selected_point])

            if self.edit_submode == "move":
                self.track_points[self.selected_point] = [x, y]
                self.recalculate_auto_angles()
            elif self.edit_submode == "rotate":
                diff = pos - p
                self.marker_angles[self.selected_point] = np.arctan2(diff[1], diff[0])

            self.update_canvas_drawings()

    def on_canvas_release(self, event):
        if not self.is_optimizing:
            self.edit_submode = None

    def toggle_ui_visibility(self):
        self.show_ui_elements = not self.show_ui_elements
        self.update_canvas_drawings()

    def update_canvas_drawings(self, *args):
        self.canvas.delete("overlay")
        if not self.show_ui_elements or len(self.track_points) == 0: return

        track_w = self.width_slider.get()
        half_w = track_w / 2.0

        for i, p in enumerate(self.track_points):
            angle = self.marker_angles[i]
            n = np.array([np.cos(angle), np.sin(angle)])
            p = np.array(p)
            pl = p + n * half_w
            pr = p - n * half_w

            selected = (i == self.selected_point and self.editor_mode.get() == "edit")

            # Farbe des Markers anpassen: Orange, wenn manuell festgesetzt, sonst Hellblau/Rot
            if selected:
                col = "#00d2ff"
            elif self.manual_angles[i]:
                col = "#ff9900"  # Orange signalisiert "Händisch fixiert"
            else:
                col = "#ff3333"

            w = 4 if selected else 2
            dot = "#00d2ff" if selected else ("#ff9900" if self.manual_angles[i] else "#0078d7")

            self.canvas.create_line(pl[0], pl[1], pr[0], pr[1], fill=col, width=w, tags="overlay")

            if selected:
                self.canvas.create_oval(pl[0] - 4, pl[1] - 4, pl[0] + 4, pl[1] + 4, fill="#ffffff", outline="#00d2ff",
                                        tags="overlay")
                self.canvas.create_oval(pr[0] - 4, pr[1] - 4, pr[0] + 4, pr[1] + 4, fill="#ffffff", outline="#00d2ff",
                                        tags="overlay")

            r = 8 if selected else 6
            self.canvas.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r,
                                    fill=dot, outline="white", width=2, tags="overlay")
            self.canvas.create_text(p[0] + 10, p[1] - 10, text=str(i + 1),
                                    fill="white", font=("Arial", 8), tags="overlay")

    # ─────────────────────────────────────────────────────────────────────────
    # SPLINE & PHYSIK
    # ─────────────────────────────────────────────────────────────────────────
    def generate_spline_line(self, centers, offsets, angles, resolution=600):
        track_w = self.width_slider.get()
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

    def calculate_lap_telemetry(self, smooth_line):
        n = len(smooth_line)
        if n < 3: return [100.0] * n, [0.0] * n

        MU = 3.5;
        G = 9.81;
        MAX_ACCEL = 320.0;
        MAX_BRAKE = 480.0
        MAX_SPEED = 520.0;
        MIN_SPEED = 55.0;
        PIXEL_SCALE = 0.35

        dists = np.zeros(n)
        for i in range(1, n):
            dists[i] = dists[i - 1] + np.linalg.norm(smooth_line[i] - smooth_line[i - 1])

        radii = np.full(n, 9999.0)
        for i in range(n):
            p1 = smooth_line[i - 1];
            p2 = smooth_line[i];
            p3 = smooth_line[(i + 1) % n]
            a = p2 - p1;
            b = p3 - p1
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
            v_next_possible = np.sqrt(speeds[i] ** 2 + 2.0 * MAX_ACCEL * seg_dist)
            speeds[ni] = min(speeds[ni], v_next_possible, MAX_SPEED)

        for _ in range(3):
            for i in range(n - 1, -1, -1):
                ni = (i + 1) % n
                seg_dist = np.linalg.norm(smooth_line[ni] - smooth_line[i])
                if seg_dist < 0.001: continue
                v_curr_possible = np.sqrt(speeds[ni] ** 2 + 2.0 * MAX_BRAKE * seg_dist)
                speeds[i] = min(speeds[i], v_curr_possible, MAX_SPEED)

        return speeds.tolist(), dists.tolist()

    def calculate_lap_time(self, smooth_line, offsets):
        if np.any(np.abs(offsets) > 0.97): return float('inf')
        speeds, _ = self.calculate_lap_telemetry(smooth_line)
        total_time = 0.0
        n = len(smooth_line)
        for i in range(n):
            seg = np.linalg.norm(smooth_line[(i + 1) % n] - smooth_line[i])
            total_time += seg / max(speeds[i], 1.0)
        return total_time

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMIERUNG
    # ─────────────────────────────────────────────────────────────────────────
    def toggle_optimization(self):
        if len(self.track_points) < 4:
            self.status_label.config(text="Min. 4 Marker nötig!", foreground="#ff4444")
            return

        if self.is_optimizing:
            self.is_optimizing = False
            self.play_btn.config(text="▶ IDEALLINIE BERECHNEN")
            self.status_label.config(text="Gestoppt", foreground="#ffaa00")
        else:
            self.is_optimizing = True
            self.generation = 0
            self.play_btn.config(text="⏸ STOPP")
            self.status_label.config(text="Optimiere...", foreground="#00ff88")

            self.centers = np.array(self.track_points, dtype=float)
            self.num_pts = len(self.centers)
            self.best_offsets = np.zeros(self.num_pts)

            self.best_line_points = self.generate_spline_line(self.centers, self.best_offsets, self.marker_angles)
            self.car_speeds, self.cum_distances = self.calculate_lap_telemetry(self.best_line_points)
            self.best_time = self.calculate_lap_time(self.best_line_points, self.best_offsets)

            self.generation_step()

    def generation_step(self):
        if not self.is_optimizing: return

        n = self.num_pts
        POPULATION = 40
        mutation_strength = max(0.02, 0.4 * (0.985 ** self.generation))
        candidates = []

        current_line = self.generate_spline_line(self.centers, self.best_offsets, self.marker_angles)
        current_time = self.calculate_lap_time(current_line, self.best_offsets)
        candidates.append((current_time, self.best_offsets.copy(), current_line))

        for _ in range(POPULATION - 1):
            test = self.best_offsets.copy()
            for k in range(n):
                if np.random.rand() < 0.4:
                    test[k] += np.random.normal(0, mutation_strength)

            test = np.clip(test, -0.95, 0.95)
            line = self.generate_spline_line(self.centers, test, self.marker_angles)
            t = self.calculate_lap_time(line, test)

            if t != float('inf') and len(line) > 2:
                vectors = np.diff(line, axis=0)
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                angle_diffs = np.abs(np.diff(angles))
                angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
                t += np.sum(angle_diffs ** 2) * 0.1

            candidates.append((t, test, line))

        candidates.sort(key=lambda x: x[0])

        if candidates[0][0] < self.best_time or self.generation == 0:
            self.best_time = candidates[0][0]
            self.best_offsets = candidates[0][1]
            self.best_line_points = candidates[0][2]
            self.car_speeds, self.cum_distances = self.calculate_lap_telemetry(self.best_line_points)

        self.generation += 1

        if self.best_line_points is not None:
            self._draw_speed_line()
            pure_time = self.calculate_lap_time(self.best_line_points, self.best_offsets)
            self.status_label.config(text=f"Zeit: {pure_time:.3f}s", foreground="#00ff88")

        self.gen_label.config(text=f"Generation: {self.generation}")
        self.root.after(10, self.generation_step)

    def _draw_speed_line(self):
        if self.best_line_points is None or len(self.car_speeds) == 0: return

        for sid in self.line_segment_ids: self.canvas.delete(sid)
        self.line_segment_ids.clear()

        pts = self.best_line_points
        speeds = np.array(self.car_speeds)
        n = len(pts)
        s_min, s_max = speeds.min(), speeds.max()

        for i in range(n):
            ni = (i + 1) % n
            col = speed_to_color(speeds[i], s_min, s_max)
            glow = self.canvas.create_line(pts[i, 0], pts[i, 1], pts[ni, 0], pts[ni, 1], fill=col, width=5,
                                           capstyle=tk.ROUND)
            self.line_segment_ids.append(glow)

        for i in range(n):
            ni = (i + 1) % n
            core = self.canvas.create_line(pts[i, 0], pts[i, 1], pts[ni, 0], pts[ni, 1], fill="white", width=1,
                                           capstyle=tk.ROUND)
            self.line_segment_ids.append(core)

        self.update_canvas_drawings()

    # ─────────────────────────────────────────────────────────────────────────
    # CAR RENDERER
    # ─────────────────────────────────────────────────────────────────────────
    def animate_car(self):
        now = time.time()
        dt = min(now - self.last_frame_time, 0.05)
        self.last_frame_time = now

        if self.car_id_body: self.canvas.delete(self.car_id_body); self.car_id_body = None
        if self.car_id_light: self.canvas.delete(self.car_id_light); self.car_id_light = None

        can_animate = (
                self.best_line_points is not None and
                len(self.best_line_points) > 10 and
                len(self.cum_distances) > 0 and
                len(self.car_speeds) > 0
        )

        if can_animate:
            total_length = self.cum_distances[-1]
            if total_length > 0:
                current_speed = float(np.interp(self.car_distance % total_length, self.cum_distances, self.car_speeds))
                self.car_distance = (self.car_distance + current_speed * dt) % total_length

                idx = np.searchsorted(self.cum_distances, self.car_distance % total_length)
                idx = min(idx, len(self.best_line_points) - 1)
                cx, cy = self.best_line_points[idx, 0], self.best_line_points[idx, 1]

                self.car_id_body = self.canvas.create_oval(
                    cx - 9, cy - 9, cx + 9, cy + 9, fill="#ccff00", outline="black", width=2)
                self.car_id_light = self.canvas.create_oval(
                    cx - 3, cy - 3, cx + 3, cy + 3, fill="#ff2200")

        self.root.after(16, self.animate_car)

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────
    def reset_only_line(self):
        self.is_optimizing = False
        self.generation = 0
        self.play_btn.config(text="▶ IDEALLINIE BERECHNEN")
        self.best_time = float('inf')
        self.best_offsets = None
        self.best_line_points = None
        self.cum_distances, self.car_speeds = [], []
        self.car_distance = 0.0
        self._draw_speed_line()
        self.status_label.config(text="Linie gelöscht", foreground="#38b6ff")
        self.gen_label.config(text="")

    def reset_track(self):
        self.is_optimizing = False
        self.show_ui_elements = True
        self.selected_point = None
        self.edit_submode = None
        self.track_points, self.marker_angles, self.manual_angles = [], [], []
        self.best_time = float('inf')
        self.best_offsets = None
        self.best_line_points = None
        self.cum_distances, self.car_speeds = [], []
        self.car_distance = 0.0
        self.generation = 0
        self.play_btn.config(text="▶ IDEALLINIE BERECHNEN")
        self.canvas.delete("overlay")
        for sid in self.line_segment_ids: self.canvas.delete(sid)
        self.line_segment_ids.clear()
        self.status_label.config(text="Bereit – Marker setzen", foreground="#888888")
        self.gen_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrackOptimizerApp(root)
    root.mainloop()
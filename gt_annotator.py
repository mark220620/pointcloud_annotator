import os
import sys
import glob
import numpy as np
import open3d as o3d

# ============================================================
# Path settings
# ============================================================
PRED_FOLDER = r"E:\pickle\point_clouds_annotator\scenes\citystreet_rainy_day_2026-05-14-09-49-14\colored_360_pcd_filter"
GT_FOLDER   = PRED_FOLDER  # GT .npz saved alongside prediction files

# Raw LiDAR PCD folder (VLS128_pcd).  Set to None to disable the left panel.
RAW_FOLDER  = os.path.join(os.path.dirname(PRED_FOLDER), "VLS128_pcd")

# ============================================================
# Mode (annotation only)
# ============================================================
# Specify a single frame. None = interactive selection.
TARGET_FRAME = None  # e.g. "000042"

# ============================================================
# Label definitions  (nuScenes lidarseg challenge — 16 classes)
#
# Index 0 is the ignore/noise class. Indices 1-16 are the 16 evaluated
# classes, following the official challenge ordering. Colors use the
# nuScenes official lidarseg colormap.
# ============================================================
LABEL_NAMES = {
    0:  "noise",
    1:  "barrier",
    2:  "bicycle",
    3:  "bus",
    4:  "car",
    5:  "construction_vehicle",
    6:  "motorcycle",
    7:  "pedestrian",
    8:  "traffic_cone",
    9:  "trailer",
    10: "truck",
    11: "driveable_surface",
    12: "other_flat",
    13: "sidewalk",
    14: "manmade",
    15: "vegetation",
    16: "terrain",
}

LABEL_COLORS = {
    0:  [  0,   0,   0],   # noise / ignore
    1:  [255, 120,  50],   # barrier
    2:  [255, 192, 203],   # bicycle
    3:  [255, 255,   0],   # bus
    4:  [  0, 150, 245],   # car
    5:  [  0, 255, 255],   # construction_vehicle
    6:  [255,  80,  80],   # motorcycle
    7:  [255,   0,   0],   # pedestrian
    8:  [255, 240, 150],   # traffic_cone
    9:  [135,  60,   0],   # trailer
    10: [160,  32, 240],   # truck
    11: [255,   0, 255],   # driveable_surface
    12: [139, 137, 137],   # other_flat
    13: [ 75,   0,  75],   # sidewalk
    14: [150, 240,  80],   # manmade
    15: [  0, 175,   0],   # vegetation
    16: [255, 200,   0],   # terrain
}

_LUT = np.array(
    [LABEL_COLORS.get(i, [128, 128, 128]) for i in range(256)], dtype=np.uint8
)

# ============================================================
# Utilities
# ============================================================

def find_pred_files():
    pattern = os.path.join(PRED_FOLDER, "*_labels.npz")
    return sorted(glob.glob(pattern))


def frame_id_from_path(path):
    return os.path.basename(path).replace("_labels.npz", "")


def gt_path_from_frame(frame_id):
    return os.path.join(GT_FOLDER, f"{frame_id}_gt.npz")


def pred_path_from_frame(frame_id):
    return os.path.join(PRED_FOLDER, f"{frame_id}_labels.npz")


def load_pred(frame_id):
    path = pred_path_from_frame(frame_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")
    data = np.load(path)
    return data["points"], data["labels"], data["colors"]


def load_gt(frame_id):
    path = gt_path_from_frame(frame_id)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["points"], data["labels"]


def load_raw(frame_id):
    """Load raw VLS128 PCD and return a tensor PointCloud coloured by intensity."""
    if RAW_FOLDER is None:
        return None
    path = os.path.join(RAW_FOLDER, f"{frame_id}.pcd")
    if not os.path.exists(path):
        return None
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float32)
    else:
        # No colour channel — derive grey from intensity if stored, else flat grey
        cols = np.full((len(pts), 3), 0.7, dtype=np.float32)
    import open3d.core as o3c
    tpcd = o3d.t.geometry.PointCloud()
    tpcd.point["positions"] = o3c.Tensor(pts[:, :3])
    tpcd.point["colors"]    = o3c.Tensor(cols)
    return tpcd


def save_gt(frame_id, points, labels):
    path = gt_path_from_frame(frame_id)
    np.savez_compressed(path, points=points, labels=labels)
    print(f"[GT saved] {path}")

    pcd_path = os.path.join(GT_FOLDER, f"{frame_id}_gt.pcd")
    pcd = make_pcd(points, labels)
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"[GT saved] {pcd_path}")


def labels_to_colors(labels):
    idx = np.clip(labels.astype(np.int32), 0, 255)
    return _LUT[idx].astype(np.float64) / 255.0


def make_pcd(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(labels_to_colors(labels))
    return pcd


def print_label_menu():
    print("\n  Label list:")
    for lbl_id, name in LABEL_NAMES.items():
        color = LABEL_COLORS[lbl_id]
        print(f"    {lbl_id:>2}: {name:<16}  RGB{tuple(color)}")


# ============================================================
# 3D paint brush (single window: rotate AND paint, Open3D gui)
# ============================================================
#
# Single-window painting built on the modern Open3D gui (o3d.visualization.gui
# + rendering), available in 0.15.1. The SceneWidget gives us a real mouse
# callback (set_on_mouse), which the legacy VisualizerWithEditing does not.
#
#   - Default left-drag       -> orbit / rotate the cloud (handled by Open3D)
#   - mouse wheel             -> zoom
#   - Shift + left-drag       -> paint: points within the brush radius (in
#                                screen space) get the target label, and the
#                                cloud recolours live so you see the result.
#   - [ / ]                   -> shrink / grow the brush
#
# Painting uses a tensor PointCloud so colors can be updated in place via
# scene.scene.update_geometry(UPDATE_COLORS_FLAG) without re-adding geometry.

SELECT_WIN_W, SELECT_WIN_H = 1800, 900
BRUSH_RADIUS_DEFAULT = 25     # screen pixels
BRUSH_RADIUS_MIN     = 4
BRUSH_RADIUS_MAX     = 200

# The Open3D gui Application is a global singleton. initialize() must be called
# exactly once per process; create_window()/run() can then be repeated. We must
# NEVER mix in the legacy Visualizer (draw_geometries) — its destructor calls
# glfwTerminate(), which kills GLFW for all subsequent gui windows.
_GUI_INITIALIZED = [False]


def _ensure_gui_app():
    """Return the gui Application, initializing it exactly once."""
    import open3d.visualization.gui as gui
    app = gui.Application.instance
    if not _GUI_INITIALIZED[0]:
        app.initialize()
        _GUI_INITIALIZED[0] = True
    return app


def _project_points_gui(points, view, proj, width, height):
    """
    Project world points to screen pixels using gui camera matrices.

    view/proj are 4x4 (world->camera, camera->clip). Returns (px, py, visible)
    where visible marks points in front of the camera. Pixel origin is top-left.
    """
    pts_h = np.hstack([points[:, :3], np.ones((len(points), 1))]).astype(np.float64)
    clip = (proj @ (view @ pts_h.T)).T            # N x 4
    w = clip[:, 3]
    visible = w > 1e-9
    w_safe = np.where(visible, w, 1.0)
    ndc_x = clip[:, 0] / w_safe
    ndc_y = clip[:, 1] / w_safe
    px = (ndc_x * 0.5 + 0.5) * width
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * height     # flip Y to screen coords
    return px, py, visible


# ============================================================
# Mode: annotate  (single persistent gui window)
# ============================================================
#
# Everything happens inside ONE gui window with ONE app.run(). We never open a
# second window or call app.run() twice — the 0.15.1 gui runloop cannot be
# restarted reliably (2nd run stops delivering key events / doesn't block).

def run_annotate(frame_id):
    import open3d.core as o3c
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    # Ordered list of all frames, so N/P can step through them.
    pred_files = find_pred_files()
    all_frames = [frame_id_from_path(p) for p in pred_files]
    if frame_id not in all_frames:
        all_frames = [frame_id] + all_frames

    print(f"\n[annotate] {len(all_frames)} frames available.")
    print_label_menu()
    print("\n  In the 3D window:")
    print("    click a swatch      pick target label (top panel)")
    print("    Shift + left-drag   paint with the target label")
    print("    left-drag / wheel   rotate / zoom")
    print("    [ / ]               shrink / grow brush")
    print("    N / P               next / previous frame (auto-saves current)")
    print("    U  undo     W  save GT     Q  save & quit     X  quit (no save)")

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0

    # 3D brush-cursor ring: a unit circle line loop we reposition on the view
    # plane to follow the mouse.
    ring_mat = rendering.MaterialRecord()
    ring_mat.shader = "unlitLine"
    ring_mat.line_width = 2.0
    _ring_n = 48
    _ring_ang = np.linspace(0, 2 * np.pi, _ring_n, endpoint=False)
    _ring_unit = np.column_stack(
        [np.cos(_ring_ang), np.sin(_ring_ang), np.zeros(_ring_n)]).astype(np.float64)
    _ring_lines = [[i, (i + 1) % _ring_n] for i in range(_ring_n)]

    app = _ensure_gui_app()
    window = app.create_window(f"Annotate — {frame_id}", SELECT_WIN_W, SELECT_WIN_H)
    em = window.theme.font_size

    def _setup_scene(w):
        """Apply common rendering settings to an Open3DScene."""
        w.scene.set_background([0.15, 0.15, 0.17, 1.0])
        w.scene.show_skybox(False)
        w.scene.scene.enable_sun_light(False)
        w.scene.scene.set_indirect_light_intensity(0.0)
        w.scene.view.set_post_processing(False)
        w.scene.view.set_antialiasing(True, False)

    # Left panel: raw LiDAR (read-only, for visual reference)
    show_raw = RAW_FOLDER is not None and os.path.isdir(RAW_FOLDER)
    widget_raw = None
    if show_raw:
        widget_raw = gui.SceneWidget()
        widget_raw.scene = rendering.Open3DScene(window.renderer)
        window.add_child(widget_raw)
        _setup_scene(widget_raw)

    # Right panel: semantic label cloud (paintable)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget)
    _setup_scene(widget)

    # ---- Top overlay panel: status line + clickable label swatches --------
    panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.25 * em, 0.5 * em, 0.25 * em))
    panel.background_color = gui.Color(0.0, 0.0, 0.0, 0.6)

    status = gui.Label("")
    status.text_color = gui.Color(1.0, 1.0, 1.0, 1.0)
    panel.add_child(status)

    # Grid of label swatches (one toggle button per label, tinted to its
    # color). Wrapped onto multiple rows so all 17 fit a 1400px window.
    SWATCHES_PER_ROW = 9
    label_buttons = {}

    def _make_label_setter(lbl):
        def _cb():
            state["label"] = lbl
            _update_panel()
            print(f"  target label -> {lbl} ({LABEL_NAMES[lbl]})")
        return _cb

    swatch_row = None
    for col, (lbl, name) in enumerate(LABEL_NAMES.items()):
        if col % SWATCHES_PER_ROW == 0:
            swatch_row = gui.Horiz(0.25 * em)
            panel.add_child(swatch_row)
        rgb = np.array(LABEL_COLORS.get(lbl, [128, 128, 128])) / 255.0
        b = gui.Button(f"{lbl}:{name}")
        b.background_color = gui.Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        b.horizontal_padding_em = 0.3
        b.vertical_padding_em = 0.1
        b.set_on_clicked(_make_label_setter(lbl))
        swatch_row.add_child(b)
        label_buttons[lbl] = b
    window.add_child(panel)

    UNDO_LIMIT = 20
    # Per-frame data lives in `state` so frame-switching just swaps it out.
    state = {
        "painting": False,
        "brush": BRUSH_RADIUS_DEFAULT,
        "label": 1,                 # current target label (persists across frames)
        "stroke": None,             # indices+old-labels for the in-progress drag
        "undo": [],                 # list of (idxs ndarray, old_labels ndarray)
        "dirty": False,             # unsaved changes for the current frame?
        "frame_id": None,
        "points": None,
        "labels": None,
        "colors": None,
        "tpcd": None,
        "scene_center": None,
    }

    def _load_frame(fid, reset_camera=True):
        """Load a frame's cloud into both panels, replacing the current one."""
        points, pred_labels, _ = load_pred(fid)
        labels = pred_labels.copy()
        gt = load_gt(fid)
        if gt is not None:
            _, gt_labels = gt
            if len(gt_labels) == len(labels):
                labels = gt_labels.copy()
                print(f"[annotate] Loaded existing GT: {gt_path_from_frame(fid)}")

        # Right panel: semantic label cloud
        colors = labels_to_colors(labels).astype(np.float32)
        tpcd = o3d.t.geometry.PointCloud()
        tpcd.point["positions"] = o3c.Tensor(points[:, :3].astype(np.float32))
        tpcd.point["colors"] = o3c.Tensor(colors)

        if widget.scene.has_geometry("pcd"):
            widget.scene.remove_geometry("pcd")
        if widget.scene.has_geometry("brush_ring"):
            widget.scene.remove_geometry("brush_ring")
        widget.scene.add_geometry("pcd", tpcd, mat)

        bounds = o3d.geometry.AxisAlignedBoundingBox(
            points[:, :3].min(axis=0), points[:, :3].max(axis=0))
        if reset_camera:
            widget.setup_camera(60.0, bounds, bounds.get_center())

        # Left panel: raw LiDAR cloud
        if show_raw and widget_raw is not None:
            raw_tpcd = load_raw(fid)
            if widget_raw.scene.has_geometry("raw_pcd"):
                widget_raw.scene.remove_geometry("raw_pcd")
            if raw_tpcd is not None:
                widget_raw.scene.add_geometry("raw_pcd", raw_tpcd, mat)
                raw_bounds = o3d.geometry.AxisAlignedBoundingBox(
                    np.asarray(raw_tpcd.point["positions"].numpy())[:, :3].min(axis=0),
                    np.asarray(raw_tpcd.point["positions"].numpy())[:, :3].max(axis=0))
                if reset_camera:
                    widget_raw.setup_camera(60.0, raw_bounds, raw_bounds.get_center())

        state.update(frame_id=fid, points=points, labels=labels, colors=colors,
                     tpcd=tpcd, scene_center=bounds.get_center(),
                     undo=[], dirty=False, stroke=None, painting=False)
        print(f"[annotate] Frame: {fid}  Points: {len(points)}")
        _update_panel()

    def _switch_frame(delta):
        """Auto-save current frame, then move to a neighbouring frame."""
        cur = state["frame_id"]
        if state["dirty"]:
            save_gt(cur, state["points"], state["labels"])
            state["dirty"] = False
        try:
            i = all_frames.index(cur)
        except ValueError:
            i = 0
        j = i + delta
        if j < 0 or j >= len(all_frames):
            print(f"  already at the {'first' if delta < 0 else 'last'} frame")
            return
        _load_frame(all_frames[j])

    def _update_panel(flash=None):
        name = LABEL_NAMES.get(state["label"], "?")
        star = " *unsaved*" if state["dirty"] else ""
        fid = state["frame_id"]
        try:
            pos = f"{all_frames.index(fid) + 1}/{len(all_frames)}"
        except ValueError:
            pos = "?"
        if flash:
            status.text = flash
        else:
            status.text = (
                f"frame {fid} [{pos}]{star}    label: {state['label']} ({name})    "
                f"brush: {state['brush']}px    undo: {len(state['undo'])}    "
                f"[N/P]frame  click swatch=label  [Shift+drag]paint  [U]ndo  [W]save  [Q]quit  [[ / ]]brush size"
            )
        # Highlight the active label swatch with a bright border-ish effect by
        # brightening only the selected one (others dimmed).
        for lbl, btn in label_buttons.items():
            rgb = np.array(LABEL_COLORS.get(lbl, [128, 128, 128])) / 255.0
            if lbl == state["label"]:
                btn.text = f"[{lbl}:{LABEL_NAMES[lbl]}]"
                btn.background_color = gui.Color(
                    float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
            else:
                btn.text = f"{lbl}:{LABEL_NAMES[lbl]}"
                btn.background_color = gui.Color(
                    float(rgb[0]) * 0.45, float(rgb[1]) * 0.45,
                    float(rgb[2]) * 0.45, 1.0)
        window.title = f"Annotate — {fid} [{pos}]{star}"
        window.post_redraw()

    def _recolor(idxs):
        colors = state["colors"]
        colors[idxs] = labels_to_colors(state["labels"][idxs]).astype(np.float32)
        state["tpcd"].point["colors"] = o3c.Tensor(colors)
        widget.scene.scene.update_geometry(
            "pcd", state["tpcd"], rendering.Scene.UPDATE_COLORS_FLAG)

    def _paint_at(mx, my):
        points, labels = state["points"], state["labels"]
        frame = widget.frame
        cam = widget.scene.camera
        view = np.array(cam.get_view_matrix(), dtype=np.float64)
        proj = np.array(cam.get_projection_matrix(), dtype=np.float64)
        px, py, visible = _project_points_gui(
            points, view, proj, frame.width, frame.height)
        lx, ly = mx - frame.x, my - frame.y
        r = state["brush"]
        d2 = (px - lx) ** 2 + (py - ly) ** 2
        hit = np.where(visible & (d2 <= r * r))[0]
        hit = hit[labels[hit] != state["label"]]
        if len(hit) == 0:
            return
        # accumulate this drag into one undo entry, recording original labels
        stroke = state["stroke"]
        new = np.setdiff1d(hit, stroke["seen"], assume_unique=False)
        if len(new):
            stroke["idxs"].append(new)
            stroke["old"].append(labels[new].copy())
            stroke["seen"].update(new.tolist())
        labels[hit] = state["label"]
        _recolor(hit)

    def _draw_cursor(mx, my):
        """Place the 3D brush ring on the view plane under the cursor (mx, my)."""
        frame = widget.frame
        cam = widget.scene.camera
        lx, ly = mx - frame.x, my - frame.y

        view = np.array(cam.get_view_matrix(), dtype=np.float64)
        right = view[0, :3]
        up = view[1, :3]
        fwd = -view[2, :3]
        cam_pos = -view[:3, :3].T @ view[:3, 3]

        # Put the ring on a plane at the scene-center depth in front of camera.
        depth = float(np.dot(state["scene_center"] - cam_pos, fwd))
        if depth <= 0 or not np.isfinite(depth):
            return
        # Physical half-height of the view frustum at that depth (fov is vertical).
        fov_v = np.radians(cam.get_field_of_view())
        half_h = np.tan(fov_v / 2.0) * depth
        world_per_px = (2.0 * half_h) / max(frame.height, 1)

        # Cursor center on that plane: start at frustum center, offset by the
        # cursor's pixel distance from the view center.
        center_world = cam_pos + fwd * depth
        dx_px = lx - frame.width / 2.0
        dy_px = ly - frame.height / 2.0
        c = (center_world
             + right * (dx_px * world_per_px)
             - up * (dy_px * world_per_px))   # screen y is down
        radius = state["brush"] * world_per_px
        if radius <= 0 or not np.isfinite(radius):
            return
        verts = c + radius * (_ring_unit[:, 0:1] * right + _ring_unit[:, 1:2] * up)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(verts)
        ls.lines = o3d.utility.Vector2iVector(_ring_lines)
        rgb = np.array(LABEL_COLORS.get(state["label"], [255, 255, 255])) / 255.0
        ls.colors = o3d.utility.Vector3dVector(np.tile(rgb, (len(_ring_lines), 1)))
        if widget.scene.has_geometry("brush_ring"):
            widget.scene.remove_geometry("brush_ring")
        widget.scene.add_geometry("brush_ring", ls, ring_mat)

    def on_layout(ctx):
        r = window.content_rect
        pref = panel.calc_preferred_size(ctx, gui.Widget.Constraints())
        panel.frame = gui.Rect(r.x, r.y, r.width, pref.height)
        top = r.y + pref.height
        h = r.height - pref.height
        if show_raw and widget_raw is not None:
            half = r.width // 2
            widget_raw.frame = gui.Rect(r.x, top, half, h)
            widget.frame     = gui.Rect(r.x + half, top, r.width - half, h)
        else:
            widget.frame = gui.Rect(r.x, top, r.width, h)
    window.set_on_layout(on_layout)

    def _sync_camera():
        """Copy right-panel camera to left-panel so both views stay in sync."""
        if not (show_raw and widget_raw is not None):
            return
        cam = widget.scene.camera
        widget_raw.scene.camera.copy_from(cam)
        widget_raw.force_redraw()

    def on_mouse(event):
        # Always keep the brush ring under the cursor on plain moves.
        if event.type == gui.MouseEvent.Type.MOVE:
            _draw_cursor(event.x, event.y)
            _sync_camera()
            return gui.SceneWidget.EventCallbackResult.IGNORED

        if not event.is_modifier_down(gui.KeyModifier.SHIFT):
            return gui.SceneWidget.EventCallbackResult.IGNORED
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            state["painting"] = True
            state["stroke"] = {"idxs": [], "old": [], "seen": set()}
            _paint_at(event.x, event.y)
            _draw_cursor(event.x, event.y)
            return gui.SceneWidget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.Type.DRAG and state["painting"]:
            _paint_at(event.x, event.y)
            _draw_cursor(event.x, event.y)
            _sync_camera()
            return gui.SceneWidget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.Type.BUTTON_UP and state["painting"]:
            state["painting"] = False
            stroke = state["stroke"]
            state["stroke"] = None
            if stroke and stroke["idxs"]:
                idxs = np.concatenate(stroke["idxs"])
                old = np.concatenate(stroke["old"])
                state["undo"].append((idxs, old))
                if len(state["undo"]) > UNDO_LIMIT:
                    state["undo"].pop(0)
                state["dirty"] = True
                _update_panel()
                print(f"  painted {len(idxs)} points as "
                      f"{state['label']} ({LABEL_NAMES[state['label']]})")
            return gui.SceneWidget.EventCallbackResult.CONSUMED
        return gui.SceneWidget.EventCallbackResult.IGNORED

    def on_key(event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.SceneWidget.EventCallbackResult.IGNORED
        k = event.key
        C = gui.SceneWidget.EventCallbackResult.CONSUMED

        if k == gui.KeyName.LEFT_BRACKET:
            state["brush"] = max(BRUSH_RADIUS_MIN, state["brush"] - 4)
            _update_panel()
            return C
        if k == gui.KeyName.RIGHT_BRACKET:
            state["brush"] = min(BRUSH_RADIUS_MAX, state["brush"] + 4)
            _update_panel()
            return C
        if k == gui.KeyName.N:
            _switch_frame(+1)
            return C
        if k == gui.KeyName.P:
            _switch_frame(-1)
            return C
        if k == gui.KeyName.U:
            if state["undo"]:
                idxs, old = state["undo"].pop()
                state["labels"][idxs] = old
                _recolor(idxs)
                state["dirty"] = True
                _update_panel()
                print(f"  undo ({len(state['undo'])} left)")
            else:
                print("  nothing to undo")
            return C
        if k == gui.KeyName.W:
            save_gt(state["frame_id"], state["points"], state["labels"])
            state["dirty"] = False
            _update_panel(flash="  Saved ✓")
            window.post_redraw()
            import threading
            def _restore():
                gui.Application.instance.post_to_main_thread(window, _update_panel)
            threading.Timer(2.0, _restore).start()
            return C
        if k == gui.KeyName.Q:
            save_gt(state["frame_id"], state["points"], state["labels"])
            window.close()
            return C
        if k == gui.KeyName.X:
            if state["dirty"]:
                dlg = gui.Dialog("Unsaved Changes")
                dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
                dlg_layout.add_child(gui.Label("You have unsaved changes.\nQuit without saving?"))
                btn_row = gui.Horiz(em)
                btn_quit = gui.Button("Quit without saving")
                btn_cancel = gui.Button("Cancel")
                def _do_quit():
                    window.close_dialog()
                    window.close()
                def _do_cancel():
                    window.close_dialog()
                btn_quit.set_on_clicked(_do_quit)
                btn_cancel.set_on_clicked(_do_cancel)
                btn_row.add_child(btn_quit)
                btn_row.add_child(btn_cancel)
                dlg_layout.add_child(btn_row)
                dlg.add_child(dlg_layout)
                window.show_dialog(dlg)
            else:
                window.close()
            return C
        return gui.SceneWidget.EventCallbackResult.IGNORED

    widget.set_on_mouse(on_mouse)
    widget.set_on_key(on_key)

    _load_frame(frame_id)   # load the starting frame
    app.run()  # single run for the whole session


# ============================================================
# Frame selection
# ============================================================

def pick_frame_interactive(pred_files):
    print("\nAvailable frames:")
    for i, path in enumerate(pred_files):
        fid = frame_id_from_path(path)
        tag = "[GT]" if os.path.exists(gt_path_from_frame(fid)) else "    "
        print(f"  {i:>3}: {fid} {tag}")
    choice = input("Enter index or frame ID: ").strip()
    if choice.isdigit() and int(choice) < len(pred_files):
        return frame_id_from_path(pred_files[int(choice)])
    return choice


# ============================================================
# Entry point
# ============================================================

def main():
    frame_id = TARGET_FRAME
    if frame_id is None:
        pred_files = find_pred_files()
        if not pred_files:
            print(f"No *_labels.npz found in {PRED_FOLDER}")
            sys.exit(1)
        frame_id = (
            frame_id_from_path(pred_files[0])
            if len(pred_files) == 1
            else pick_frame_interactive(pred_files)
        )

    run_annotate(frame_id)


if __name__ == "__main__":
    main()

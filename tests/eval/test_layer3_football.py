import cv2
import numpy as np
import pytest
from eval.layer3_football.ball import detect_ball_candidates
from eval.layer3_football.pitch_geometry import extract_pitch_lines
from eval.layer3_football.goal_net import detect_goal_candidate_roi


def test_ball_integrity_pitch_line_suppression():
    """Verify that pitch line fragments are suppressed and genuine balls are detected."""
    h, w = 300, 400
    # Green pitch (BGR: [35, 120, 35])
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = [35, 120, 35]

    # Draw white pitch line: touchline across the pitch
    cv2.line(frame, (30, 80), (370, 80), (255, 255, 255), thickness=4)

    # Draw a genuine circular football on the pitch away from the line
    ball_center = (200, 200)
    cv2.circle(frame, ball_center, radius=8, color=(255, 255, 255), thickness=-1)

    lines = extract_pitch_lines(frame)
    assert len(lines) >= 1

    # Detect candidates with pitch line suppression
    candidates = detect_ball_candidates(frame, pitch_lines=lines, min_circularity=0.50)

    # There should be at least 1 ball candidate at (200, 200)
    assert len(candidates) >= 1
    ball_found = False
    for cx, cy, r, circ in candidates:
        if abs(cx - 200) <= 5 and abs(cy - 200) <= 5:
            ball_found = True
        # Ensure no candidates lie along the pitch line y=80
        assert abs(cy - 80) > 8, f"False ball detected on pitch line at ({cx}, {cy})"

    assert ball_found is True


def test_pitch_geometry_perspective_clustering():
    """Verify that orthogonal pitch markings do not trigger false wobble events."""
    # Orthogonal markings: touchlines around 0.05 rad, cross lines around 1.6 rad
    # Bundle A (touchlines)
    lines_a = [
        (10, 50, 300, 55),
        (10, 100, 300, 108),
        (10, 150, 300, 160),
    ]
    # Bundle B (perpendicular lines)
    lines_b = [
        (50, 10, 55, 200),
        (150, 10, 158, 200),
    ]
    all_lines = lines_a + lines_b

    # Cluster angles
    angles = [(np.arctan2(y2 - y1, x2 - x1) % np.pi) for x1, y1, x2, y2 in all_lines]
    cl_a = [angles[0]]
    cl_b = []
    wobble_events = 0
    for ang in angles[1:]:
        d_a = min(abs(ang - cl_a[0]) % np.pi, np.pi - abs(ang - cl_a[0]) % np.pi)
        if d_a < 0.45:
            cl_a.append(ang)
        elif not cl_b or min(abs(ang - cl_b[0]) % np.pi, np.pi - abs(ang - cl_b[0]) % np.pi) < 0.45:
            cl_b.append(ang)
        else:
            wobble_events += 1

    for cl in (cl_a, cl_b):
        if len(cl) >= 3 and float(np.std(cl)) > 0.35:
            wobble_events += 1

    # Standard deviation within each bundle is low -> zero wobble events
    assert wobble_events == 0
    assert len(cl_a) >= 2
    assert len(cl_b) >= 2


def test_goal_net_presence_gating():
    """Verify that midfield scene without goal net defaults to clean 5.0 and ROI is None."""
    # Midfield scene: grass with white kit player
    h, w = 200, 300
    midfield_frame = np.zeros((h, w, 3), dtype=np.uint8)
    midfield_frame[:, :] = [35, 120, 35]

    # Outfield player wearing white kit
    cv2.rectangle(midfield_frame, (140, 90), (160, 150), (240, 240, 240), -1)

    roi = detect_goal_candidate_roi(midfield_frame)
    # Midfield frame should NOT detect a goal ROI
    assert roi is None


def test_broadcast_graphics_persistence_check():
    """Verify that static graphics with high persistence are distinguished from transient edges."""
    h, w = 100, 200
    # Simulate top-left scoreboard: static white rectangle across 10 frames
    edge_acc = np.zeros((h, w), dtype=np.float32)
    diffs = []
    for _ in range(10):
        frame_edge = np.zeros((h, w), dtype=np.float32)
        # Static scoreboard border
        frame_edge[10:30, 10:80] = 1.0
        edge_acc += frame_edge
        diffs.append(0.5)

    mean_edge = edge_acc / 10.0
    # Scoreboard pixels have 100% persistence >= 60%
    persistent_pixels = np.sum(mean_edge >= 0.60)
    assert persistent_pixels >= 40

    # Transient crowd movement: random edge noise
    transient_acc = np.zeros((h, w), dtype=np.float32)
    np.random.seed(42)
    for _ in range(10):
        rand_edges = (np.random.rand(h, w) > 0.90).astype(np.float32)
        transient_acc += rand_edges

    mean_transient = transient_acc / 10.0
    persistent_transient = np.sum(mean_transient >= 0.60)
    # Transient movement rarely reaches 60% temporal persistence on same pixels
    assert persistent_transient < 20


def test_curved_markings_do_not_trigger_pitch_wobble():
    """Verify that circular pitch markings (center circle, arcs) are not misclassified as wobble/bent lines."""
    from eval.layer3_football.pitch_geometry import evaluate_pitch_geometry

    # Create green pitch frame with straight touchline + center circle
    h, w = 400, 600
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = [35, 120, 35]

    # Draw straight touchline (length = 500px >= 70px)
    cv2.line(frame, (50, 100), (550, 100), (255, 255, 255), thickness=4)
    # Draw straight halfway line (length = 200px >= 70px)
    cv2.line(frame, (300, 100), (300, 300), (255, 255, 255), thickness=4)

    # Draw center circle (radius = 60px -> will create short tangent chords ~40px)
    cv2.circle(frame, (300, 200), 60, (255, 255, 255), thickness=4)

    # Run across 5 identical static frames
    score, details = evaluate_pitch_geometry([frame] * 5)
    assert details["wobble_events"] == 0
    assert score >= 4.5


def test_duplicate_ball_requires_active_tracking():
    """Verify that multiple low-circularity candidate blobs without an active tracked ball do NOT trigger phantom duplicates."""
    from eval.layer3_football.ball import evaluate_ball_integrity

    h, w = 200, 300
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = [35, 120, 35]  # Green pitch

    # Draw two elongated non-ball blobs (circularity < 0.50, e.g. white shoes/rectangles)
    cv2.rectangle(frame, (50, 50), (75, 55), (255, 255, 255), -1)
    cv2.rectangle(frame, (90, 50), (115, 55), (255, 255, 255), -1)

    score, details = evaluate_ball_integrity([frame] * 4, source_has_ball=False)
    assert details["duplicate_ball_events"] == 0


def test_evaluate_ball_triplet_clean():
    """Verify that a genuine linearly moving ball in triplet (A, X, B) passes with clean metrics."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    fx = np.zeros((h, w, 3), dtype=np.uint8)
    fx[:, :] = [35, 120, 35]
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]

    # Ball moves from (100, 100) -> (110, 100) -> (120, 100)
    cv2.circle(fa, (100, 100), 8, (255, 255, 255), -1)
    cv2.circle(fx, (110, 100), 8, (255, 255, 255), -1)
    cv2.circle(fb, (120, 100), 8, (255, 255, 255), -1)

    res = evaluate_ball_triplet(fa, fx, fb)
    assert res["ball_active"] is True
    assert res["ball_matched"] is True
    assert res["ball_dissolved"] is False
    assert res["ball_wobble"] is False
    assert res["ball_deformed"] is False
    assert res["ball_duplicate"] is False


def test_evaluate_ball_triplet_dissolved():
    """Verify that ball dissolution in intermediate frame FX is flagged."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    fx = np.zeros((h, w, 3), dtype=np.uint8)
    fx[:, :] = [35, 120, 35]  # No ball in FX!
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]

    cv2.circle(fa, (100, 100), 8, (255, 255, 255), -1)
    cv2.circle(fb, (120, 100), 8, (255, 255, 255), -1)

    res = evaluate_ball_triplet(fa, fx, fb)
    assert res["ball_active"] is True
    assert res["ball_matched"] is False
    assert res["ball_dissolved"] is True


def test_evaluate_ball_triplet_wobble():
    """Verify that non-collinear ball positioning in FX triggers wobble."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    fx = np.zeros((h, w, 3), dtype=np.uint8)
    fx[:, :] = [35, 120, 35]
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]

    # FA at (100, 100), FB at (140, 100), but FX is displaced down to (120, 118) (perp dist = 18px > 8px)
    cv2.circle(fa, (100, 100), 8, (255, 255, 255), -1)
    cv2.circle(fx, (120, 118), 8, (255, 255, 255), -1)
    cv2.circle(fb, (140, 100), 8, (255, 255, 255), -1)

    res = evaluate_ball_triplet(fa, fx, fb)
    assert res["ball_active"] is True
    assert res["ball_matched"] is True
    assert res["ball_wobble"] is True


def test_evaluate_ball_triplet_offscreen():
    """Verify that off-screen scenes with no ball in FA/FB do not penalize FX."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    fx = np.zeros((h, w, 3), dtype=np.uint8)
    fx[:, :] = [35, 120, 35]
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]

    # Non-ball noise blob in FX
    cv2.rectangle(fx, (100, 100), (115, 108), (255, 255, 255), -1)

    res = evaluate_ball_triplet(fa, fx, fb)
    assert res["ball_active"] is False
    assert res["ball_dissolved"] is False


def test_evaluate_pitch_geometry_triplet_clean_and_circle_invariance():
    """Verify that linear pitch lines and circular markings yield low residual and zero warp events."""
    from eval.layer3_football.pitch_geometry import evaluate_pitch_geometry_triplet

    h, w = 400, 600
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    fx = fa.copy()
    fb = fa.copy()

    # Draw touchline and center circle on all 3 frames
    for f in (fa, fx, fb):
        cv2.line(f, (50, 100), (550, 100), (255, 255, 255), thickness=4)
        cv2.circle(f, (300, 200), 70, (255, 255, 255), thickness=4)

    res = evaluate_pitch_geometry_triplet(fa, fx, fb)
    assert res["pitch_active"] is True
    assert res["line_residual"] < 0.005
    assert res["warp_events"] == 0


def test_evaluate_pitch_geometry_triplet_synthetic_warp():
    """Verify that an interpolated frame with a bent/curved pitch line (when sources are straight) is flagged."""
    from eval.layer3_football.pitch_geometry import evaluate_pitch_geometry_triplet

    h, w = 400, 600
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    fb = fa.copy()
    fx = fa.copy()

    # Straight touchlines in source frames FA and FB
    cv2.line(fa, (50, 200), (550, 200), (255, 255, 255), thickness=6)
    cv2.line(fb, (50, 200), (550, 200), (255, 255, 255), thickness=6)

    # FX has a wavy/bent touchline
    pts = []
    for x in range(50, 550, 5):
        y = int(200 + 15 * np.sin((x - 50) / 40.0))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(fx, pts[i], pts[i + 1], (255, 255, 255), thickness=6)

    res = evaluate_pitch_geometry_triplet(fa, fx, fb)
    assert res["pitch_active"] is True
    assert res["warp_events"] >= 1 or res["line_residual"] > 0.01


def test_evaluate_ball_triplet_ghost_duplicate_trajectory_aligned():
    """Verify that duplicate ghost ball is only flagged when aligned with motion trajectory."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]

    # Ball moves from (100, 100) to (160, 100)
    cv2.circle(fa, (100, 100), 8, (255, 255, 255), -1)
    cv2.circle(fb, (160, 100), 8, (255, 255, 255), -1)

    # Scenario 1: FX has primary ball at (130, 100) and a ghost duplicate along trajectory at (112, 100)
    fx_ghost = np.zeros((h, w, 3), dtype=np.uint8)
    fx_ghost[:, :] = [35, 120, 35]
    cv2.circle(fx_ghost, (130, 100), 8, (255, 255, 255), -1)
    cv2.circle(fx_ghost, (112, 100), 8, (255, 255, 255), -1)
    res_ghost = evaluate_ball_triplet(fa, fx_ghost, fb)
    assert res_ghost["ball_duplicate"] is True

    # Scenario 2: FX has primary ball at (130, 100) and an off-trajectory blob at (130, 150) (e.g. shoe)
    fx_clean = np.zeros((h, w, 3), dtype=np.uint8)
    fx_clean[:, :] = [35, 120, 35]
    cv2.circle(fx_clean, (130, 100), 8, (255, 255, 255), -1)
    cv2.circle(fx_clean, (130, 150), 8, (255, 255, 255), -1)
    res_clean = evaluate_ball_triplet(fa, fx_clean, fb)
    assert res_clean["ball_duplicate"] is False


def test_evaluate_ball_triplet_hybrid_scene_cut_to_crowd():
    """Verify that a camera cut from pitch play to crowd/viewer emotions does not penalize FX."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    # FA is green pitch with ball
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    cv2.circle(fa, (100, 100), 8, (255, 255, 255), -1)

    # FX and FB are crowd shots (fans in stands with no green pitch)
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [180, 50, 50]  # Red stadium seats
    fx = fb.copy()

    res = evaluate_ball_triplet(fa, fx, fb)
    assert res["ball_active"] is False
    assert res["ball_dissolved"] is False
    assert res["ball_duplicate"] is False


def test_evaluate_ball_triplet_hybrid_cut_from_crowd_to_match():
    """Verify that a cut from viewer emotion/crowd back to live pitch does not penalize FX."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    # FA is crowd shot
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [50, 50, 180]

    # FB is green pitch with ball
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]
    cv2.circle(fb, (200, 150), 8, (255, 255, 255), -1)

    fx = fb.copy()

    res = evaluate_ball_triplet(fa, fx, fb)
    assert res["ball_active"] is False
    assert res["ball_dissolved"] is False


def test_evaluate_ball_triplet_hybrid_camera_cut_different_ball_positions():
    """Verify that a cut between two different camera shots with ball at different positions is not penalized as teleportation."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    cv2.circle(fa, (50, 50), 8, (255, 255, 255), -1)

    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]
    # Ball is far away at (350, 250) (distance ~ 360px > 160px)
    cv2.circle(fb, (350, 250), 8, (255, 255, 255), -1)

    fx = fa.copy()

    res = evaluate_ball_triplet(fa, fx, fb)
    assert res["ball_active"] is False
    assert res["ball_dissolved"] is False


def test_evaluate_ball_triplet_natural_player_occlusion():
    """Verify that when a ball travels behind a player's body in FX, it is recognized as natural occlusion rather than dissolution."""
    from eval.layer3_football.ball import evaluate_ball_triplet

    h, w = 300, 400
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]
    cv2.circle(fa, (100, 100), 8, (255, 255, 255), -1)

    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [35, 120, 35]
    cv2.circle(fb, (140, 100), 8, (255, 255, 255), -1)

    # In FX, expected pos is (120, 100). A player is standing at bbox (105, 70, 30, 60)
    fx = np.zeros((h, w, 3), dtype=np.uint8)
    fx[:, :] = [35, 120, 35]
    players_x = [{"bbox": (105, 70, 30, 60), "solidity": 0.90}]

    res = evaluate_ball_triplet(fa, fx, fb, players_x=players_x)
    assert res["ball_active"] is True
    assert res["ball_matched"] is False
    assert res["ball_occluded"] is True
    assert res["ball_dissolved"] is False


def test_evaluate_pitch_geometry_triplet_hybrid_cut_to_crowd():
    """Verify that pitch geometry triplet is inactive when cutting to crowd/viewer emotions."""
    from eval.layer3_football.pitch_geometry import evaluate_pitch_geometry_triplet

    h, w = 400, 600
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [35, 120, 35]  # Pitch with touchline
    cv2.line(fa, (50, 100), (550, 100), (255, 255, 255), thickness=4)

    # FB is crowd shot with no green pitch
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [50, 60, 200]
    fx = fb.copy()

    res = evaluate_pitch_geometry_triplet(fa, fx, fb)
    assert res["pitch_active"] is False
    assert res["warp_events"] == 0
    assert res["line_residual"] == 0.0




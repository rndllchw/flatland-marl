"""Exploratory (E8 exploratory 2, videos for the slides): one T1 scenario played by three policies.

T1 scenario 08, evaluation seed 2, no disruptions, sampled actions: MAPPO s1 (all 10 trains arrive), IPPO s1
(trains 2 and 5 get lost) and the starter-kit checkpoint (5 trains jam). These illustrate the failure types found in
E8 exploratory 2; they are not evidence that one arm is better (on this scenario train 5 also gets lost under IPPO
s0, IPPO s2 and MAPPO s0). Each replay must reproduce the stored E8 result (arrived, penalty per train) or the script
stops.

The map is flatland's renderer, drawn once. On top, every step: each train as a dot in its own colour (the same in
every clip) and its destination as a square in that colour; trains that end lost (still driving, drove at least their
shortest route; buckets from results/e8_explore/failures_trains.csv) get a red ring and a red trail, trains that end
stuck an orange ring. Arrived trains leave the map.

Writes results/e8_explore/videos/: one mp4 per policy (20 frames/s, one frame per step, last frame held 2 s),
side_by_side.mp4 (2 x 2: the three clips and a legend) and lost_trails.png (IPPO's last frame, full trails).

    python scripts/make_videos.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool
from PIL import Image, ImageDraw, ImageFont

from flatmarl.envs import clear_flatland_caches, load_scenario
from flatmarl.metrics import episode_metrics
from flatmarl.policies import make_policy

sys.path.insert(0, str(Path(__file__).parent))
import explore_failures as ef  # noqa: E402  (policy names, stored E8 results, output folder)

SCENARIO, SEED = "08", 2
CLIPS = {"MAPPO s1": "mappo_s1", "IPPO s1": "ippo_s1", "checkpoint": "checkpoint"}  # label -> file name
FPS, HOLD = 20, 2 * 20           # frames per second (one step per frame); frames to hold the last frame
TILE = (960, 768)                # one clip in the 2 x 2 video
OUT = ef.OUT / "videos"
PALETTE = [(31, 119, 180), (44, 160, 44), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (0, 0, 128), (0, 128, 128)]  # no red or orange: kept for lost / stuck
RING = {"lost": (220, 20, 20), "stuck": (255, 140, 0)}


def replay(label):
    """Play the episode again -> (per step: each train's (cell or None, arrived)), destinations, max steps.
    Stops unless every train's arrived / penalty equals the stored E8 result."""
    name, folder, _ = ef.POLICIES[label]
    policy, obs_builder = make_policy(ef.runnable(name), seed=SEED)
    env = load_scenario(ef.TEST_SET / f"{SCENARIO}.pkl", disruption="none", seed=SEED, obs_builder=obs_builder)
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    steps = []

    def snapshot():
        steps.append([(a.position, a.state == TrainState.DONE) for a in env.agents])

    snapshot()
    while True:
        obs, _, dones, _ = env.step(policy.act_many(handles, [obs[h] for h in handles]))
        snapshot()
        if dones["__all__"]:
            break
    replayed = [(t.arrived, t.penalty) for t in episode_metrics(env).trains]
    targets, max_steps = [a.target for a in env.agents], env._max_episode_steps
    clear_flatland_caches()

    stored = pd.read_csv(ef.STORED / folder / "trains.csv", dtype={"scenario": str})
    stored = stored[(stored.policy == name) & (stored.scenario == SCENARIO) & (stored.seed == SEED)]
    if replayed != list(zip(stored.arrived, stored.penalty)):
        raise SystemExit(f"{label}: replay differs from the stored E8 result; nothing written")
    return steps, targets, max_steps


def outcomes(label) -> dict:
    """train -> "lost" / "stuck" for the trains of this episode that did not arrive (E8 exploratory 2 buckets)."""
    f = pd.read_csv(ef.OUT / "failures_trains.csv", dtype={"scenario": str})
    f = f[(f.label == label) & (f.scenario == SCENARIO) & (f.seed == SEED)]
    lost = (f.bucket == "still_driving") & (f.driven >= f.shortest_route)
    return {**{t: "stuck" for t in f[f.bucket == "stuck"].train}, **{t: "lost" for t in f[lost].train}}


def background():
    """Flatland's picture of the map (no trains) and its cell size in pixels."""
    env = load_scenario(ef.TEST_SET / f"{SCENARIO}.pkl", disruption="none", seed=SEED)
    env.reset()
    renderer = RenderTool(env, gl="PILSVG", screen_width=1800, screen_height=1440)
    renderer.render_env(show=False, show_agents=False, show_observations=False, show_predictions=False,
                        show_rowcols=False)
    image, cell = Image.fromarray(renderer.get_image()[:, :, :3]), renderer.gl.nPixCell
    clear_flatland_caches()
    return image, cell


def draw_map(base, cell, steps, step, targets, ends):
    """The map at `step` (held at the last step once the episode is over) with trains, destinations and trails."""
    img = base.copy()
    d = ImageDraw.Draw(img)
    r = 1.6 * cell

    def centre(rc):
        return rc[1] * cell + cell / 2, rc[0] * cell + cell / 2

    for t, target in enumerate(targets):
        x, y = centre(target)
        d.rectangle([x - r, y - r, x + r, y + r], outline=PALETTE[t % len(PALETTE)], width=4)
    for t, end in ends.items():
        trail = [centre(s[t][0]) for s in steps[:step + 1] if s[t][0] is not None]
        if end == "lost" and len(trail) > 1:
            d.line(trail, fill=RING["lost"], width=3)
    for t, (cell_rc, _) in enumerate(steps[step]):
        if cell_rc is None:  # not departed yet, or arrived
            continue
        x, y = centre(cell_rc)
        if t in ends:
            d.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], outline=RING[ends[t]], width=5)
        d.ellipse([x - r, y - r, x + r, y + r], fill=PALETTE[t % len(PALETTE)], outline=(255, 255, 255), width=2)
    return img


def header(img, text, size):
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=size)
    left, top, right, bottom = d.textbbox((12, 8), text, font=font)
    d.rectangle([left - 8, top - 6, right + 8, bottom + 6], fill=(255, 255, 255))
    d.text((12, 8), text, fill=(0, 0, 0), font=font)
    return img


def legend_tile():
    img = Image.new("RGB", TILE, (255, 255, 255))
    d = ImageDraw.Draw(img)
    big, small = ImageFont.load_default(size=30), ImageFont.load_default(size=22)
    d.text((40, 40), "Test set T1, scenario 08, 10 trains", fill=(0, 0, 0), font=big)
    d.text((40, 85), "same map and timetable for all three; no breakdowns; sampled actions", fill=(80, 80, 80),
           font=small)
    rows = [("dot", PALETTE[0], "a train (own colour, same in every clip)"),
            ("square", PALETTE[0], "its destination"),
            ("ring", RING["lost"], "ends LOST: drives in a loop, never arrives (red trail)"),
            ("ring", RING["stuck"], "ends STUCK: stands still for its final 50+ steps"),
            (None, None, "arrived trains leave the map")]
    y = 160
    for kind, colour, text in rows:
        if kind == "dot":
            d.ellipse([50, y, 78, y + 28], fill=colour, outline=(255, 255, 255))
        elif kind == "square":
            d.rectangle([50, y, 78, y + 28], outline=colour, width=4)
        elif kind == "ring":
            d.ellipse([46, y - 4, 82, y + 32], outline=colour, width=5)
            d.ellipse([52, y + 2, 76, y + 26], fill=PALETTE[0])
        d.text((105, y), text, fill=(0, 0, 0), font=small)
        y += 60
    d.text((40, y + 40), "Illustration, not evidence: on this scenario train 5 also gets lost", fill=(80, 80, 80),
           font=small)
    d.text((40, y + 72), "under 4 of the 6 trained networks. Rates: EXPERIMENTS.md, E8 exploratory 2.",
           fill=(80, 80, 80), font=small)
    return img


class Video:
    """Frames piped straight into ffmpeg (H.264), so no frame files are written."""

    def __init__(self, path: Path, size):
        self.path = path
        self.proc = subprocess.Popen(
            [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "23", str(path)], stdin=subprocess.PIPE)

    def write(self, img):
        self.proc.stdin.write(np.asarray(img.convert("RGB")).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise SystemExit(f"ffmpeg failed for {self.path}")


def main():
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found (macOS: brew install ffmpeg)")
    episodes = {label: replay(label) for label in CLIPS}
    print("replays match the stored E8 results:",
          ", ".join(f"{label} {sum(a for _, a in steps[-1])}/{len(targets)} arrived"
                    for label, (steps, targets, _) in episodes.items()))
    ends = {label: outcomes(label) for label in CLIPS}
    base, cell = background()
    OUT.mkdir(parents=True, exist_ok=True)
    singles = {label: Video(OUT / f"{name}.mp4", base.size) for label, name in CLIPS.items()}
    grid = Video(OUT / "side_by_side.mp4", (2 * TILE[0], 2 * TILE[1]))
    legend = legend_tile()
    n = max(len(steps) for steps, _, _ in episodes.values())
    for frame in range(n + HOLD):
        tiles = []
        for label, (steps, targets, max_steps) in episodes.items():
            step = min(frame, len(steps) - 1)
            img = draw_map(base, cell, steps, step, targets, ends[label])
            text = f"{label}   step {step} / {max_steps}   arrived {sum(a for _, a in steps[step])} / {len(targets)}"
            if frame < len(steps) + HOLD:
                singles[label].write(header(img.copy(), text, 34))
            tiles.append(header(img.resize(TILE, Image.LANCZOS), text, 24))
            if label == "IPPO s1" and frame == len(steps) - 1:
                header(img.copy(), text, 34).save(OUT / "lost_trails.png")
        canvas = Image.new("RGB", (2 * TILE[0], 2 * TILE[1]))
        for i, tile in enumerate([*tiles, legend]):
            canvas.paste(tile, ((i % 2) * TILE[0], (i // 2) * TILE[1]))
        grid.write(canvas)
    for video in [*singles.values(), grid]:
        video.close()
    for path in sorted(OUT.iterdir()):
        print(f"wrote {path.relative_to(ef.ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
import os

import numpy as np

import evaluate as E
from go1_env import FAULT_TYPES, JOINT_NAMES, Go1Batch

DIRS = {"forward 0.6": (0.6, 0, 0), "backward 0.4": (-0.4, 0, 0), "left 0.3": (0, 0.3, 0),
        "right 0.3": (0, -0.3, 0), "turn 0.8": (0, 0, 0.8), "stand": (0, 0, 0)}
FAULTS = {"healthy": None, "limp FR thigh": ("limp", "FR_thigh"), "limp FL thigh": ("limp", "FL_thigh"),
          "limp RR thigh": ("limp", "RR_thigh"), "limp RL thigh": ("limp", "RL_thigh"),
          "limp FR calf": ("limp", "FR_calf"), "weak FL thigh": ("weak", "FL_thigh"),
          "frozen RR calf": ("frozen", "RR_calf")}


def rollout(pol, cmd, fault, n=16, steps=750, onset=100, seed=0):
    env = Go1Batch(n, seed=seed, randomize=True, obs_noise=True, command_mode="fixed", push_max=0.5, **pol.env_kwargs)
    env.fixed_command = np.array(cmd, float)
    env.forced_fault = None if fault is None else (JOINT_NAMES.index(fault[1]), FAULT_TYPES.index(fault[0]), onset)
    for i in range(n):
        env._reset_env(i)
    a, _ = env.observe()
    pol.reset()
    alive = np.ones(n, bool)
    V, W, C, Z = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    done = None
    for t in range(steps):
        a, _, _, done, _, _ = env.step(pol.step(a, done))
        for i in range(n):
            if not alive[i]:
                continue
            if done[i]:
                alive[i] = False
                continue
            if t >= onset + 50:
                d = env.datas[i]
                R = d.xmat[env.trunk_id].reshape(3, 3)
                V[i].append((R.T @ d.qvel[0:3])[:2])
                W[i].append(d.qvel[5])
                C[i].append(env._true_contact(d))
                Z[i].append(d.geom_xpos[env.foot_gids, 2])
    ok = [i for i in range(n) if len(V[i]) > 20]
    if not ok:
        return dict(fall_rate=float(1 - alive.mean()))
    v = np.array([np.mean(V[i], 0) for i in ok])
    w = np.array([np.mean(W[i]) for i in ok])
    lift = np.median([np.median(np.max(Z[i], 0) - np.median(Z[i], 0)) for i in ok]) * 1000
    return dict(fall_rate=float(1 - alive.mean()), vx=float(v[:, 0].mean()), vy=float(v[:, 1].mean()),
                yaw=float(w.mean()), lin_err=float(np.linalg.norm(v - np.array(cmd[:2]), axis=1).mean()),
                yaw_err=float(np.abs(w - cmd[2]).mean()),
                on_floor=float(np.mean([np.mean(C[i]) for i in ok])), lift_mm=float(lift))


def video(path, out, fault=("limp", "FR_thigh"), seed=2, panels_spec=None):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import imageio
    import mujoco
    from PIL import Image, ImageDraw
    script = [(0, "forward", (0.6, 0, 0)), (4, "left", (0, 0.3, 0)), (8, "backward", (-0.4, 0, 0)),
              (12, "right", (0, -0.3, 0)), (16, "turn", (0, 0, 0.8)), (20, "stand", (0, 0, 0))]
    W_, H_ = 480, 360
    panels = []
    spec = panels_spec or [(path, None, "healthy"), (path, fault, f"{fault[0]} {fault[1]} from t=2s")]
    plain_cmd = {"forward": "walk forward", "left": "step left", "backward": "walk backward",
                 "right": "step right", "turn": "turn on the spot", "stand": "stand still"}
    try:
        from PIL import ImageFont
        f_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
        f_med = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        f_big = f_med = None
    for item in spec:
        ppath, flt, plabel = item[:3]
        fcap = item[3] if len(item) > 3 else None
        pol = E.load(ppath)
        env = Go1Batch(1, seed=seed, randomize=False, obs_noise=True, command_mode="fixed", push_max=0.0,
                       visual=True, **pol.env_kwargs)
        env.forced_fault = None if flt is None else (JOINT_NAMES.index(flt[1]), FAULT_TYPES.index(flt[0]), 100)
        env._reset_env(0)
        env.terminate = False
        m, d = env.models[0], env.datas[0]
        d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        a, _ = env.observe()
        pol.reset()
        r = mujoco.Renderer(m, H_, W_)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.trunk_id
        cam.distance, cam.elevation, cam.azimuth = 1.8, -22, 135
        frames, fell_at = [], None
        for s in range(int(24 / 0.02)):
            tsec = s * 0.02
            label, cmd = [(n, c) for t0, n, c in script if tsec >= t0][-1]
            env.fixed_command = np.array(cmd, float)
            env.cmd[0] = env.fixed_command
            a, _, _, _, _, terms = env.step(pol.step(a))
            if fell_at is None and terms[0, 2] > 0:
                fell_at = tsec
            if s % 2 == 0:
                r.update_scene(d, cam)
                img = Image.fromarray(r.render())
                dr = ImageDraw.Draw(img, "RGBA")
                if fcap is None:
                    dr.text((10, 8), plabel, fill=(255, 255, 255))
                    dr.text((10, 24), f"t={tsec:4.1f}s  command: {label} {cmd}", fill=(255, 220, 120))
                    if fell_at is not None:
                        dr.text((10, 40), f"fall detected at t={fell_at:.1f}s", fill=(255, 110, 110))
                else:
                    lines = [(plabel, f_big, (255, 255, 255)), (fcap, f_med, (215, 225, 240)),
                             (f"Command: {plain_cmd[label]}", f_med, (255, 220, 120))]
                    if fell_at is not None:
                        lines.append((f"Fell at {fell_at:.1f} s", f_big, (255, 120, 120)))
                    dr.rectangle([0, 0, W_, 12 + 21 * len(lines)], fill=(0, 0, 0, 120))
                    for li, (txt, fnt, col) in enumerate(lines):
                        dr.text((10, 6 + 21 * li), txt, fill=col, font=fnt)
                    dr.text((W_ - 70, 8), f"{tsec:4.1f} s", fill=(200, 200, 200), font=f_med)
                frames.append(np.asarray(img))
        r.close()
        panels.append(frames)
    combo = [np.concatenate([p[k] for p in panels], axis=1) for k in range(min(map(len, panels)))]
    imageio.mimwrite(out, combo, fps=25, quality=7, macro_block_size=8)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=[])
    ap.add_argument("--out", default="results/omni_eval.json")
    ap.add_argument("--video", default="")
    ap.add_argument("--faults", nargs="*", default=None, help="subset of FAULTS names to run")
    args = ap.parse_args()
    if args.video:
        video(args.video, "results/part4_all_directions.mp4")
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for p in args.policies:
        name = os.path.basename(os.path.dirname(p))
        pol = E.load(p)
        res.setdefault(name, {})
        for fi, (fname, flt) in enumerate(FAULTS.items()):
            if args.faults and fname not in args.faults:
                continue
            for di, (dname, cmd) in enumerate(DIRS.items()):
                r = rollout(pol, cmd, flt, seed=1000 + 10 * fi + di)
                res[name][f"{fname} | {dname}"] = r
                print(f"{name:14s} {fname:15s} {dname:13s} " + " ".join(
                    f"{kk}={vv:.2f}" for kk, vv in r.items()), flush=True)
            json.dump(res, open(args.out, "w"), indent=2)

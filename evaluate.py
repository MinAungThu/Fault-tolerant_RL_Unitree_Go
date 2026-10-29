import argparse
import json
import os

import numpy as np
import torch

from go1_env import ACTOR_DIM, CRITIC_DIM, FAULT_TYPES, JOINT_NAMES, Go1Batch
from fault_classifier import LongHistory, make_fc_actor_critic
from ppo import ActorCritic

EVAL_STEPS = 750
ONSET = 100


class Policy:

    def __init__(self, ac, env_kwargs=None):
        self.ac = ac
        self.fc = hasattr(ac, "clf")
        self.env_kwargs = env_kwargs or {}
        self.hist = LongHistory()

    def reset(self):
        self.hist = LongHistory()

    def __call__(self, a_obs, long_obs=None):
        a_obs = torch.as_tensor(a_obs)
        return self.ac.act_deterministic(a_obs, long_obs) if self.fc else self.ac.act_deterministic(a_obs)

    def step(self, a_obs, reset=None):
        long_obs = self.hist.update(a_obs, reset)
        return self(a_obs, long_obs).numpy()


def load(path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    a = ck.get("args", {})
    env_kwargs = {}
    if a.get("clf"):
        ac = make_fc_actor_critic()
    elif a.get("gait_clock"):
        ac = ActorCritic(actor_dim=ACTOR_DIM + 2, critic_dim=CRITIC_DIM + 2)
        env_kwargs = {"gait_clock": True}
    else:
        ac = ActorCritic()
    ac.load_state_dict(ck["model"])
    ac.eval()
    return Policy(ac, env_kwargs)


def run_condition(ac, fault, n_envs=8, seed=0, push_max=0.0, randomize=False):
    env = Go1Batch(n_envs, seed=seed, randomize=randomize, obs_noise=True, command_mode="fixed", push_max=push_max,
                   **ac.env_kwargs)
    env.forced_fault = fault
    for i in range(n_envs):
        env._reset_env(i)
    a_obs, _ = env.observe()
    ac.reset()
    alive = np.ones(n_envs, bool)
    vx = [[] for _ in range(n_envs)]
    done = None
    for t in range(EVAL_STEPS):
        act = ac.step(a_obs, done)
        a_obs, _, _, done, _, terms = env.step(act)
        for i in range(n_envs):
            if alive[i]:
                if done[i]:
                    alive[i] = False
                else:
                    vx[i].append(terms[i, 5])
    after = [np.mean(v[ONSET + 50:]) if len(v) > ONSET + 50 else 0.0 for v in vx]
    dist = [np.sum(v) * 0.02 for v in vx]
    return dict(fall_rate=float(1 - alive.mean()), vx_after_fault=float(np.mean(after)),
                distance=float(np.mean(dist)))


def evaluate(paths, out_dir, n_envs=8, push_max=0.0, randomize=False):
    conds = [("none", None)] + [(f"{FAULT_TYPES[t]}:{JOINT_NAMES[j]}", (j, t, ONSET))
                               for t in range(1, len(FAULT_TYPES)) for j in range(12)]
    results = {}
    for p in paths:
        name = os.path.basename(os.path.dirname(p))
        ac = load(p)
        results[name] = {}
        for k, (cname, fault) in enumerate(conds):
            results[name][cname] = run_condition(ac, fault, n_envs=n_envs, seed=k, push_max=push_max, randomize=randomize)
            r = results[name][cname]
            print(f"{name:12s} {cname:18s} fall={r['fall_rate']:.2f} vx={r['vx_after_fault']:.2f} dist={r['distance']:.1f}", flush=True)
    json.dump(results, open(os.path.join(out_dir, "eval_results.json"), "w"), indent=2)
    return results


def summary_table(results):
    lines = ["| policy | condition | fall rate | forward speed after fault (m/s, cmd 0.6) | distance in 15 s (m) |",
             "|---|---|---|---|---|"]
    for name, res in results.items():
        groups = {"healthy": ["none"]}
        for t in FAULT_TYPES[1:]:
            groups[t] = [k for k in res if k.startswith(t + ":")]
        for g, keys in groups.items():
            fr = np.mean([res[k]["fall_rate"] for k in keys])
            vx = np.mean([res[k]["vx_after_fault"] for k in keys])
            di = np.mean([res[k]["distance"] for k in keys])
            lines.append(f"| {name} | {g} | {fr:.2f} | {vx:.2f} | {di:.1f} |")
    return "\n".join(lines)


def plot(results, out_dir, fname="fault_survival.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    colors = ["#9aa4b2", "#2a6fdb", "#e3a33b", "#1f9e74", "#c2456b"]
    for ax, t in zip(axes, FAULT_TYPES[1:]):
        x = np.arange(12)
        w = 0.8 / len(names)
        for k, n in enumerate(names):
            surv = [1 - results[n][f"{t}:{j}"]["fall_rate"] for j in JOINT_NAMES]
            ax.bar(x + (k - (len(names) - 1) / 2) * w, surv, w, label=n, color=colors[k % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, rotation=60, fontsize=8)
        ax.set_title(f"{t} motor (from t = 2 s)")
        ax.set_ylim(0, 1.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("survival rate over 15 s")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(names), loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out_dir, fname), dpi=140)


def video(paths, out_dir, fault=("limp", "FR_calf"), seconds=10, push=False, seed=3,
          labels=None, clf_overrides=None, fname=None):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import imageio
    import mujoco
    from PIL import Image, ImageDraw

    j, t = JOINT_NAMES.index(fault[1]), FAULT_TYPES.index(fault[0])
    W, H = 480, 360
    frames_all = []
    for k, p in enumerate(paths):
        ac = load(p)
        if clf_overrides and clf_overrides[k] is not None:
            from checks import FixedLogits
            ac.ac.clf = FixedLogits(clf_overrides[k])
        env = Go1Batch(1, seed=seed, randomize=False, obs_noise=True, command_mode="fixed",
                       push_max=0.6 if push else 0.0, visual=True, **ac.env_kwargs)
        env.forced_fault = (j, t, ONSET)
        env._reset_env(0)
        m, d = env.models[0], env.datas[0]
        d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        a_obs, _ = env.observe()
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.trunk_id
        cam.distance, cam.elevation, cam.azimuth = 1.6, -18, 140
        env.terminate = False
        ac.reset()
        frames, fell_at = [], None
        for s in range(int(seconds / 0.02)):
            act = ac.step(a_obs)
            a_obs, _, _, _, _, terms = env.step(act)
            if fell_at is None and terms[0, 2] > 0:
                fell_at = s * 0.02
            if s % 2 == 0:
                r.update_scene(d, cam)
                img = Image.fromarray(r.render())
                dr = ImageDraw.Draw(img)
                label = labels[k] if labels else os.path.basename(os.path.dirname(p))
                status = f"{fault[0]} {fault[1]}" if env.fault_active[0] else "healthy"
                dr.text((10, 8), f"{label}", fill=(255, 255, 255))
                dr.text((10, 24), f"t={s*0.02:4.1f}s  {status}", fill=(255, 220, 120))
                if fell_at is not None:
                    dr.text((10, 40), f"fall detected at t={fell_at:.1f}s", fill=(255, 110, 110))
                frames.append(np.asarray(img))
        r.close()
        frames_all.append(frames)
    n = min(len(f) for f in frames_all)
    combo = [np.concatenate([f[k] for f in frames_all], axis=1) for k in range(n)]
    fn = os.path.join(out_dir, fname or f"compare_{fault[0]}_{fault[1]}{'_push' if push else ''}.mp4")
    imageio.mimwrite(fn, combo, fps=25, quality=7, macro_block_size=8)
    print("wrote", fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--robust", action="store_true", help="16 robots/condition, domain randomisation + 0.5 m/s pushes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.skip_eval:
        res = evaluate(args.policies, args.out, **(dict(n_envs=16, push_max=0.5, randomize=True) if args.robust else {}))
        open(os.path.join(args.out, "summary.md"), "w").write(summary_table(res) + "\n")
        print(summary_table(res))
        plot(res, args.out)
    if args.video:
        for f in [("limp", "FR_calf"), ("frozen", "RL_thigh"), ("weak", "FL_thigh")]:
            video(args.policies, args.out, f)

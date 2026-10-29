import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from go1_env import ACTOR_DIM, FAULT_TYPES, JOINT_NAMES, OBS_DIM

LONG_HIST = 25
LONG_DIM = LONG_HIST * OBS_DIM
N_CLASSES = 1 + 3 * 12
FAULT_SLICE = slice(OBS_DIM + 3 + 1, OBS_DIM + 3 + 1 + 36)


def class_name(c):
    if c == 0:
        return "healthy"
    t, j = (c - 1) // 12 + 1, (c - 1) % 12
    return f"{FAULT_TYPES[t]}:{JOINT_NAMES[j]}"


def labels_from_critic(c_obs):
    f = torch.as_tensor(c_obs)[:, FAULT_SLICE]
    return torch.where(f.max(-1).values > 0.5, f.argmax(-1) + 1, torch.zeros(f.shape[0], dtype=torch.long))


class LongHistory:

    def __init__(self):
        self.h = None

    def update(self, a_obs, reset=None):
        new = torch.as_tensor(a_obs)[:, -OBS_DIM:]
        if self.h is None or self.h.shape[0] != new.shape[0]:
            self.h = new[:, None].repeat(1, LONG_HIST, 1).clone()
        else:
            self.h = torch.cat([self.h[:, 1:], new[:, None]], dim=1)
            if reset is not None:
                r = torch.as_tensor(reset, dtype=torch.bool)
                if r.any():
                    self.h[r] = new[r][:, None].repeat(1, LONG_HIST, 1)
        return self.h.reshape(new.shape[0], -1)


class FaultClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(LONG_DIM))
        self.register_buffer("std", torch.ones(LONG_DIM))
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(LONG_DIM, 512), nn.ELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.ELU(), nn.Linear(256, N_CLASSES))

    def forward(self, x):
        return self.net(torch.clamp((x - self.mean) / self.std, -5, 5))


def make_fc_actor_critic(base_state=None, clf_state=None):
    from ppo import ActorCritic, mlp

    class ActorCriticFC(ActorCritic):
        def __init__(self):
            super().__init__()
            self.clf = FaultClassifier()
            self.actor = mlp(ACTOR_DIM + N_CLASSES, 12)

        def _feat(self, actor_obs, long_obs):
            p = F.softmax(self.clf(long_obs), -1).detach()
            return torch.cat([self.actor_norm(actor_obs), p], -1)

        def dist(self, actor_obs, long_obs):
            mu = self.actor(self._feat(actor_obs, long_obs))
            return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

        @torch.no_grad()
        def act_deterministic(self, actor_obs, long_obs):
            return self.actor(self._feat(actor_obs, long_obs))

    ac = ActorCriticFC()
    if base_state is not None:
        own = ac.state_dict()
        for k, v in base_state.items():
            if k == "actor.0.weight" and v.shape[1] == ACTOR_DIM:
                w = torch.zeros_like(own[k])
                w[:, :ACTOR_DIM] = v
                own[k] = w
            elif k in own and own[k].shape == v.shape:
                own[k] = v
        ac.load_state_dict(own)
    if clf_state is not None:
        ac.clf.load_state_dict(clf_state)
    return ac


def collect(policy_path, n_steps, workers, envs_per_worker, keep_every, seed):
    from evaluate import load
    from ppo import FOCUS_FAULTS, REAR_FAULTS, VecEnv
    pol = load(policy_path)
    env = VecEnv(workers, envs_per_worker, seed, fault_prob=0.85, push_max=0.5,
                 focus_faults=FOCUS_FAULTS + REAR_FAULTS, focus_prob=0.5)
    a_obs, c_obs = env.observe()
    hist = LongHistory()
    done = None
    since = np.zeros(env.n, np.int64)
    last_lab = np.zeros(env.n, np.int64)
    X, Y, S, E = [], [], [], []
    rng = np.random.default_rng(seed)
    for t in range(n_steps):
        long_obs = hist.update(a_obs, done)
        lab = labels_from_critic(c_obs).numpy()
        since = np.where(lab == last_lab, since + 1, 0)
        last_lab = lab
        if t % keep_every == 0 and t >= LONG_HIST:
            X.append(long_obs.numpy().astype(np.float16))
            Y.append(lab.copy())
            S.append(since.copy())
            E.append(np.arange(env.n))
        act = pol(torch.as_tensor(a_obs), long_obs) + rng.normal(0, 0.1, (env.n, 12)).astype(np.float32)
        a_obs, c_obs, _, done, _, _ = env.step(act)
        if done.any():
            since[done] = 0
            last_lab[done] = 0
    env.close()
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), np.concatenate(E)


def leg_of(c):
    c = np.asarray(c)
    return np.where(c == 0, 0, 1 + ((c - 1) % 12) // 3)


@torch.no_grad()
def predict(clf, X):
    was = clf.training
    clf.eval()
    p = torch.cat([clf(torch.as_tensor(X[i:i + 4096]).float()).argmax(-1) for i in range(0, len(X), 4096)]).numpy()
    clf.train(was)
    return p


def report(clf, X, Y, S, prefix=""):
    pred = predict(clf, X)
    out = {}
    late = S >= LONG_HIST
    out["acc_all"] = float((pred == Y).mean())
    out["acc_faulted_after_0.5s"] = float((pred == Y)[(Y > 0) & late].mean())
    out["healthy_false_alarm_rate"] = float((pred != 0)[Y == 0].mean())
    out["fault_missed_rate_after_0.5s"] = float((pred == 0)[(Y > 0) & late].mean())
    out["leg_acc_faulted_after_0.5s"] = float((leg_of(pred) == leg_of(Y))[(Y > 0) & late].mean())
    out["per_type_acc_after_0.5s"] = {FAULT_TYPES[t]: float((pred == Y)[(Y > 0) & late & ((Y - 1) // 12 + 1 == t)].mean())
                                      for t in range(1, 4)}
    thighs = {n: 1 + (1 - 1) * 12 + JOINT_NAMES.index(n) for n in ("FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh")}
    conf = {}
    for n, c in thighs.items():
        m = (Y == c) & late
        if m.sum() == 0:
            continue
        p = pred[m]
        conf[f"limp {n}"] = {"n": int(m.sum()), "correct": float((p == c).mean()),
                            "front_thigh": float(np.isin(p, [thighs['FR_thigh'], thighs['FL_thigh']]).mean()),
                            "rear_thigh": float(np.isin(p, [thighs['RR_thigh'], thighs['RL_thigh']]).mean()),
                            "healthy": float((p == 0).mean())}
    out["limp_thigh_confusion_after_0.5s"] = conf
    print(prefix + json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="runs/student_style_bc/latest.pt")
    ap.add_argument("--out", default="runs/fault_clf")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--envs_per_worker", type=int, default=32)
    ap.add_argument("--keep_every", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    t0 = time.time()
    X, Y, S, E = collect(args.policy, args.steps, args.workers, args.envs_per_worker, args.keep_every, args.seed)
    print(f"collected {len(X)} windows in {time.time() - t0:.0f}s; "
          f"faulted fraction {np.mean(Y > 0):.2f}", flush=True)
    n_env = args.workers * args.envs_per_worker
    test = E >= int(n_env * 0.8)
    val = (E >= int(n_env * 0.65)) & ~test
    train = ~test & ~val
    clf = FaultClassifier()
    Xtr = X[train].astype(np.float32)
    clf.mean.copy_(torch.as_tensor(Xtr.mean(0)))
    clf.std.copy_(torch.as_tensor(Xtr.std(0) + 1e-3))
    del Xtr
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    idx_tr = np.where(train)[0]
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        np.random.shuffle(idx_tr)
        tot = 0.0
        for i in range(0, len(idx_tr), 512):
            b = idx_tr[i:i + 512]
            loss = F.cross_entropy(clf(torch.as_tensor(X[b]).float()), torch.as_tensor(Y[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        v_acc = float((predict(clf, X[val]) == Y[val]).mean())
        if v_acc > best:
            best, best_state = v_acc, {k: v.clone() for k, v in clf.state_dict().items()}
        print(f"epoch {ep + 1}: train CE {tot / len(idx_tr):.3f}  val acc {v_acc:.3f}", flush=True)
    clf.load_state_dict(best_state)
    res = report(clf, X[test], Y[test], S[test], prefix="held-out robots: ")
    torch.save(clf.state_dict(), os.path.join(args.out, "clf.pt"))
    json.dump(res, open(os.path.join(args.out, "pretrain_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from go1_env import ACTOR_DIM, CRITIC_DIM, JOINT_NAMES, OBS_DIM, Go1Batch

FOCUS_FAULTS = [(JOINT_NAMES.index("FR_thigh"), 1), (JOINT_NAMES.index("FL_thigh"), 1)]
FOCUS_IDX = [OBS_DIM + 3 + 1 + (t - 1) * 12 + j for j, t in FOCUS_FAULTS]
REAR_FAULTS = [(JOINT_NAMES.index("RR_thigh"), 1), (JOINT_NAMES.index("RL_thigh"), 1)]
REAR_IDX = [OBS_DIM + 3 + 1 + (t - 1) * 12 + j for j, t in REAR_FAULTS]


def _worker(remote, kwargs):
    env = Go1Batch(**kwargs)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "observe":
            remote.send(env.observe())
        elif cmd == "curriculum":
            env.set_curriculum(**data)
            remote.send(None)
        elif cmd == "stats":
            remote.send(env.pop_stats())
        elif cmd == "close":
            remote.close()
            break


class VecEnv:
    def __init__(self, num_workers, envs_per_worker, seed, **kw):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(num_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, dict(num_envs=envs_per_worker, seed=seed * 1000 + w, **kw)), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.n = num_workers * envs_per_worker

    def _all(self, cmd, data=None):
        for r in self.remotes:
            r.send((cmd, data))
        return [r.recv() for r in self.remotes]

    def observe(self):
        res = self._all("observe")
        return np.concatenate([x[0] for x in res]), np.concatenate([x[1] for x in res])

    def step(self, actions):
        chunks = np.array_split(actions, len(self.remotes))
        for r, a in zip(self.remotes, chunks):
            r.send(("step", a))
        res = [r.recv() for r in self.remotes]
        return tuple(np.concatenate([x[k] for x in res]) for k in range(6))

    def curriculum(self, **kw):
        self._all("curriculum", kw)

    def stats(self):
        return sum(self._all("stats"), [])

    def close(self):
        for r in self.remotes:
            r.send(("close", None))


class RunningNorm(nn.Module):
    def __init__(self, dim, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(i, o, hidden=(512, 256, 128)):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, init_std=0.5, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM):
        super().__init__()
        self.actor_norm = RunningNorm(actor_dim)
        self.critic_norm = RunningNorm(critic_dim)
        self.actor = mlp(actor_dim, 12)
        self.critic = mlp(critic_dim, 1)
        self.log_std = nn.Parameter(torch.full((12,), float(np.log(init_std))))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, actor_obs):
        mu = self.actor(self.actor_norm(actor_obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, critic_obs):
        return self.critic(self.critic_norm(critic_obs)).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, actor_obs):
        return self.actor(self.actor_norm(actor_obs))


def load_widened(ac, state, n_front_actor=0, n_back_critic=0):
    own = ac.state_dict()
    for k, v in state.items():
        if k not in own:
            continue
        t = own[k]
        if t.shape == v.shape:
            own[k] = v
        elif k == "actor.0.weight":
            t.zero_()
            t[:, n_front_actor:] = v
        elif k == "critic.0.weight":
            t.zero_()
            t[:, :v.shape[1]] = v
        elif k in ("actor_norm.mean", "actor_norm.var"):
            t.fill_(0.0 if k.endswith("mean") else 1.0)
            t[n_front_actor:] = v
        elif k in ("critic_norm.mean", "critic_norm.var"):
            t.fill_(0.0 if k.endswith("mean") else 1.0)
            t[:v.shape[0]] = v
        else:
            raise ValueError(f"cannot widen {k}: {tuple(v.shape)} -> {tuple(t.shape)}")
    ac.load_state_dict(own)
    return ac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="fault_aware")
    p.add_argument("--faults", action="store_true", help="train with motor faults + pushes (curriculum)")
    p.add_argument("--steps", type=float, default=20e6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--envs_per_worker", type=int, default=32)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.005)
    p.add_argument("--kl_target", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--stop_at", type=int, default=-1,
                   help="stop after this iteration (curriculum still follows --steps); use with --resume to train in chunks")
    p.add_argument("--init", default="", help="start from these weights (fresh optimiser, no curriculum ramp)")
    p.add_argument("--teacher", action="store_true",
                   help="specialist teacher: every episode gets a limp thigh (see --teacher_faults)")
    p.add_argument("--teacher_faults", default="front", choices=["front", "rear"],
                   help="which limp thighs the --teacher run trains on")
    p.add_argument("--anchor_teacher", default="",
                   help="policy to imitate on every state NOT covered by a specialist teacher (keeps old skills)")
    p.add_argument("--anchor_coef", type=float, default=0.2)
    p.add_argument("--focus_prob", type=float, default=0.0,
                   help="fraction of faulted episodes that get a limp front thigh (oversampling)")
    p.add_argument("--style_teacher", default="", help="teacher checkpoint for the AcL-style reward")
    p.add_argument("--style_coef", type=float, default=1.0)
    p.add_argument("--bc_coef", type=float, default=0.0,
                   help="extra supervised loss pulling the student's mean action toward the teacher on focus states")
    p.add_argument("--rear_teacher", default="",
                   help="second teacher used on limp REAR thigh states (e.g. the original fault_aware policy)")
    p.add_argument("--clf", default="",
                   help="pretrained fault classifier (fault_classifier.py); its class probabilities are fed to the actor")
    p.add_argument("--clf_coef", type=float, default=1.0)
    p.add_argument("--clf_lr", type=float, default=3e-4)
    p.add_argument("--cmd_profile", default="random", choices=["random", "omni"],
                   help="'random' = original forward-biased commands (Parts 1-3); 'omni' = balanced directions (Part 4)")
    p.add_argument("--track_sigma", type=float, default=0.25, help="velocity-tracking reward kernel width")
    p.add_argument("--gait_clock", action="store_true",
                   help="give the actor a 2 Hz gait clock and reward trot timing when healthy and moving (Part 4)")
    p.add_argument("--phase_coef", type=float, default=0.5)
    p.add_argument("--gait_rewards", action="store_true",
                   help="real foot contact + foot-slip penalty + 0.2 s air-time threshold (fixes the Parts 1-3 shuffle)")
    p.add_argument("--slip_coef", type=float, default=0.2)
    p.add_argument("--std_anneal", default="",
                   help="'start,end,frac': force the exploration std from start to end over the first frac of "
                        "training (then hold). Makes the training policy close to the noise-free one we deploy.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    out = os.path.join("runs", args.run)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)

    if args.teacher:
        focus_faults = FOCUS_FAULTS if args.teacher_faults == "front" else REAR_FAULTS
    else:
        focus_faults = FOCUS_FAULTS + (REAR_FAULTS if args.rear_teacher else [])
    run_focus_idx = [OBS_DIM + 3 + 1 + (t_ - 1) * 12 + j_ for j_, t_ in focus_faults]
    focus = dict(focus_faults=focus_faults, focus_prob=1.0 if args.teacher else args.focus_prob)
    fp0 = (1.0 if args.teacher else 0.7) if args.init else 0.0
    env = VecEnv(args.workers, args.envs_per_worker, args.seed, fault_prob=fp0, push_max=0.0,
                 command_mode=args.cmd_profile, track_sigma=args.track_sigma,
                 gait_clock=args.gait_clock, phase_coef=args.phase_coef,
                 gait_rewards=args.gait_rewards, slip_coef=args.slip_coef, **focus)
    N, T = env.n, args.horizon
    n_clock = 2 if args.gait_clock else 0
    A_DIM, C_DIM = ACTOR_DIM + n_clock, CRITIC_DIM + n_clock
    use_clf = bool(args.clf)
    if use_clf and args.gait_clock:
        raise ValueError("--clf and --gait_clock are not combined in this code")
    if use_clf:
        from fault_classifier import LONG_DIM, LongHistory, labels_from_critic, make_fc_actor_critic
        base = torch.load(args.init, weights_only=False)["model"] if args.init else None
        ac = make_fc_actor_critic(base, torch.load(args.clf, weights_only=False))
        ac.clf.eval()
        main_params = [p_ for n_, p_ in ac.named_parameters() if not n_.startswith("clf.")]
        clf_opt = torch.optim.Adam(ac.clf.parameters(), lr=args.clf_lr)
        long_hist = LongHistory()
    else:
        ac = ActorCritic(actor_dim=A_DIM, critic_dim=C_DIM)
        if args.init:
            load_widened(ac, torch.load(args.init, weights_only=False)["model"], n_front_actor=n_clock, n_back_critic=n_clock)
        main_params = list(ac.parameters())
    opt = torch.optim.Adam(main_params, lr=args.lr)
    lr = args.lr
    start_it, total = 0, 0

    def load_teacher(path):
        ck_ = torch.load(path, weights_only=False)
        clk = bool(ck_.get("args", {}).get("gait_clock"))
        t_ = ActorCritic(actor_dim=ACTOR_DIM + 2 * clk, critic_dim=CRITIC_DIM + 2 * clk)
        t_.load_state_dict(ck_["model"])
        if clk and not args.gait_clock:
            raise ValueError(f"teacher {path} needs a gait clock but this run has none")
        return t_.eval(), clk

    def teacher_act(t_, clk, obs):
        return t_.act_deterministic(obs if clk else obs[:, -ACTOR_DIM:])

    teachers = []
    if args.style_teacher:
        teachers.append((*load_teacher(args.style_teacher), FOCUS_IDX))
    if args.rear_teacher:
        teachers.append((*load_teacher(args.rear_teacher), REAR_IDX))
    anchor = load_teacher(args.anchor_teacher) if args.anchor_teacher else None
    if args.resume:
        ck = torch.load(args.resume, weights_only=False)
        ac.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if use_clf and "clf_opt" in ck:
            clf_opt.load_state_dict(ck["clf_opt"])
        start_it, total, lr = ck["it"], ck["steps"], ck["lr"]

    def pdist(a_, l_):
        return ac.dist(a_, l_) if use_clf else ac.dist(a_)

    n_iters = int(args.steps // (N * T))
    a_obs, c_obs = map(torch.as_tensor, env.observe())
    log_f = open(os.path.join(out, "log.csv"), "a")
    if start_it == 0:
        log_f.write("it,steps,time,mean_reward,ep_len,fall_rate,lin_track,v_err,fault_prob,push_max,std,lr,kl,"
                    "style,focus_frac,clf_acc,clf_false_alarm,anchor_mse\n")
    ep_lens, falls = [], []
    t0 = time.time()
    prev_done = None

    last_it = n_iters if args.stop_at < 0 else min(n_iters, args.stop_at)
    for it in range(start_it, last_it):
        frac = it / n_iters
        if args.init:
            fp, pm, pen = (1.0, 0.5, 1.0) if args.teacher else (0.7, 0.8, 1.0)
        elif args.faults:
            fp = float(np.clip((frac - 0.15) / 0.35, 0.0, 1.0)) * 0.7
            pm = float(np.clip((frac - 0.10) / 0.30, 0.0, 1.0)) * 0.8
        else:
            fp, pm = 0.0, float(np.clip((frac - 0.10) / 0.30, 0.0, 1.0)) * 0.8
        if not args.init:
            pen = 0.2 + 0.8 * float(np.clip(frac / 0.4, 0.0, 1.0))
        env.curriculum(fault_prob=fp, push_max=pm, pen_scale=pen)
        if args.std_anneal:
            s0, s1, sf = map(float, args.std_anneal.split(","))
            with torch.no_grad():
                ac.log_std.fill_(float(np.log(s0 + (s1 - s0) * min(frac / sf, 1.0))))

        buf_a = torch.zeros(T, N, A_DIM)
        buf_c = torch.zeros(T, N, C_DIM)
        buf_act = torch.zeros(T, N, 12)
        buf_logp = torch.zeros(T, N)
        buf_v = torch.zeros(T, N)
        buf_r = torch.zeros(T, N)
        buf_d = torch.zeros(T, N)
        buf_teach = torch.zeros(T, N, 12)
        buf_mask = torch.zeros(T, N, dtype=torch.bool)
        buf_anchor = torch.zeros(T, N, 12) if anchor is not None else None
        buf_long = torch.zeros(T, N, LONG_DIM) if use_clf else None
        terms_acc = np.zeros(6)
        style_sum, focus_cnt = 0.0, 0
        with torch.no_grad():
            for t in range(T):
                ac.actor_norm.update(a_obs)
                ac.critic_norm.update(c_obs)
                long_obs = long_hist.update(a_obs, prev_done) if use_clf else None
                dist = pdist(a_obs, long_obs)
                act = dist.sample()
                v = ac.value(c_obs)
                na, nc, r, d, to, terms = env.step(act.numpy())
                prev_done = d
                r = torch.as_tensor(r)
                for teach, t_clk, idx_list in teachers:
                    m_ = c_obs[:, idx_list].max(-1).values > 0.5
                    focus_cnt += int(m_.sum())
                    if not m_.any():
                        continue
                    a_t = teacher_act(teach, t_clk, a_obs[m_])
                    a_s = act[m_]
                    r_scale = torch.exp(-torch.norm(a_s - a_t, dim=-1))
                    cos = (a_s * a_t).sum(-1) / (a_s.norm(dim=-1) * a_t.norm(dim=-1) + 1e-6)
                    r_style = args.style_coef * (r_scale + 0.5 * (cos + 1.0) / 2.0) * 0.02
                    r[m_] += r_style
                    style_sum += float(r_style.sum())
                    buf_teach[t, m_] = a_t
                    buf_mask[t] |= m_
                if not teachers:
                    focus_cnt += int((c_obs[:, run_focus_idx].max(-1).values > 0.5).sum())
                if anchor is not None:
                    buf_anchor[t] = teacher_act(anchor[0], anchor[1], a_obs)
                to_t = torch.as_tensor(to, dtype=torch.float32)
                r = r + args.gamma * v * to_t
                if use_clf:
                    buf_long[t] = long_obs
                buf_a[t], buf_c[t], buf_act[t] = a_obs, c_obs, act
                buf_logp[t], buf_v[t], buf_r[t] = dist.log_prob(act).sum(-1), v, r
                buf_d[t] = torch.as_tensor(d, dtype=torch.float32)
                terms_acc += terms.mean(0)
                a_obs, c_obs = torch.as_tensor(na), torch.as_tensor(nc)
            last_v = ac.value(c_obs)

        adv = torch.zeros(T, N)
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nonterm = 1.0 - buf_d[t]
            delta = buf_r[t] + args.gamma * nv * nonterm - buf_v[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_v

        fa, fc, fact = buf_a.reshape(-1, A_DIM), buf_c.reshape(-1, C_DIM), buf_act.reshape(-1, 12)
        fadv, fret = adv.reshape(-1), ret.reshape(-1)
        fteach, fmask = buf_teach.reshape(-1, 12), buf_mask.reshape(-1)
        fanchor = buf_anchor.reshape(-1, 12) if anchor is not None else None
        anchor_mse = 0.0
        flong = buf_long.reshape(-1, LONG_DIM) if use_clf else None
        flab = labels_from_critic(fc) if use_clf else None
        with torch.no_grad():
            flogp = pdist(fa, flong).log_prob(fact).sum(-1)
            fv = ac.value(fc)
            clf_acc = clf_fa = float("nan")
            if use_clf:
                pred = ac.clf(flong).argmax(-1)
                clf_acc = float((pred == flab).float().mean())
                h = flab == 0
                clf_fa = float((pred[h] != 0).float().mean()) if h.any() else float("nan")
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        B = fa.shape[0]
        mb = B // args.minibatches
        kl_mean = 0.0
        for _ in range(args.epochs):
            perm = torch.randperm(B)
            for k in range(args.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                dist = pdist(fa[idx], flong[idx] if use_clf else None)
                logp = dist.log_prob(fact[idx]).sum(-1)
                ratio = torch.exp(logp - flogp[idx])
                s1 = ratio * fadv[idx]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * fadv[idx]
                pol_loss = -torch.min(s1, s2).mean()
                v = ac.value(fc[idx])
                v_clip = fv[idx] + torch.clamp(v - fv[idx], -args.clip, args.clip)
                v_loss = torch.max((v - fret[idx]) ** 2, (v_clip - fret[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pol_loss + 1.0 * v_loss - args.ent * ent
                if args.bc_coef > 0:
                    m = fmask[idx]
                    if m.any():
                        loss = loss + args.bc_coef * ((dist.mean[m] - fteach[idx][m]) ** 2).sum(-1).mean()
                if anchor is not None:
                    am = ~fmask[idx]
                    if am.any():
                        a_loss = ((dist.mean[am] - fanchor[idx][am]) ** 2).sum(-1).mean()
                        loss = loss + args.anchor_coef * a_loss
                        anchor_mse += float(a_loss.detach())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(main_params, 1.0)
                opt.step()
                if use_clf:
                    ac.clf.train()
                    ce = F.cross_entropy(ac.clf(flong[idx]), flab[idx])
                    ac.clf.eval()
                    clf_opt.zero_grad()
                    (args.clf_coef * ce).backward()
                    nn.utils.clip_grad_norm_(ac.clf.parameters(), 1.0)
                    clf_opt.step()
                with torch.no_grad():
                    kl = (flogp[idx] - logp).mean().item()
                    kl_mean += kl
                    if kl > 2 * args.kl_target:
                        lr = max(1e-5, lr / 1.5)
                    elif kl < args.kl_target / 2:
                        lr = min(1e-2, lr * 1.5)
                    for g in opt.param_groups:
                        g["lr"] = lr
            with torch.no_grad():
                ac.log_std.clamp_(np.log(0.05), np.log(1.5))
        kl_mean /= args.epochs * args.minibatches

        total += B
        for s in env.stats():
            ep_lens.append(s["len"])
            falls.append(s["fell"])
        ep_lens, falls = ep_lens[-200:], falls[-200:]
        tm = terms_acc / T
        line = (f"{it},{total},{time.time() - t0:.0f},{buf_r.mean().item() / 0.02:.4f},"
                f"{np.mean(ep_lens) if ep_lens else 0:.1f},{np.mean(falls) if falls else 0:.3f},"
                f"{tm[0]:.3f},{tm[3]:.3f},{fp:.2f},{pm:.2f},{ac.log_std.exp().mean().item():.3f},{lr:.2e},{kl_mean:.4f},"
                f"{style_sum / max(focus_cnt, 1) / 0.02:.3f},{focus_cnt / (N * T):.3f},{clf_acc:.3f},{clf_fa:.3f},"
                f"{anchor_mse / (args.epochs * args.minibatches):.4f}")
        log_f.write(line + "\n")
        log_f.flush()
        if it % 20 == 0:
            sps = (it - start_it + 1) * B / (time.time() - t0)
            print(f"[{args.run}] it {it}/{n_iters}  {line}  sps={sps:.0f}", flush=True)
        if it % 100 == 0 or it == last_it - 1:
            ck = dict(model=ac.state_dict(), opt=opt.state_dict(), it=it + 1, steps=total, lr=lr, args=vars(args))
            if use_clf:
                ck["clf_opt"] = clf_opt.state_dict()
            torch.save(ck, os.path.join(out, "latest.pt"))
            if it % 500 == 0:
                torch.save(ck, os.path.join(out, f"ckpt_{it:05d}.pt"))
    env.close()


if __name__ == "__main__":
    main()

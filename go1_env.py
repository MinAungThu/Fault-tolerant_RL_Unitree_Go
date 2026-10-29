import copy
import os

import mujoco
import numpy as np

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_go1", "scene.xml")

JOINT_NAMES = [f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
FAULT_TYPES = ("none", "limp", "frozen", "weak")

CTRL_DT = 0.02
SIM_DT = 0.004
DECIM = int(round(CTRL_DT / SIM_DT))
EP_LEN = 1000
HIST = 5
OBS_DIM = 45
ACTOR_DIM = OBS_DIM * HIST
CRITIC_DIM = OBS_DIM + 3 + 1 + 3 * 12 + 3 + 4
ACTION_SCALE = 0.25
TROT_OFFSETS = np.array([0.0, 0.5, 0.5, 0.0])


OMNI_MODES = {
    "forward":  (0.25, lambda r: [r.uniform(0.2, 1.0), r.uniform(-0.2, 0.2), r.uniform(-0.6, 0.6)]),
    "backward": (0.15, lambda r: [r.uniform(-0.8, -0.2), r.uniform(-0.2, 0.2), r.uniform(-0.4, 0.4)]),
    "sideways": (0.15, lambda r: [r.uniform(-0.2, 0.2), r.choice([-1, 1]) * r.uniform(0.15, 0.5), r.uniform(-0.3, 0.3)]),
    "turn":     (0.15, lambda r: [0.0, 0.0, r.choice([-1, 1]) * r.uniform(0.4, 1.0)]),
    "mixed":    (0.20, lambda r: [r.uniform(-0.6, 1.0), r.uniform(-0.5, 0.5), r.uniform(-1.0, 1.0)]),
    "stand":    (0.10, lambda r: [0.0, 0.0, 0.0]),
}


def sample_omni_command(rng):
    names = list(OMNI_MODES)
    k = rng.choice(len(names), p=[OMNI_MODES[n][0] for n in names])
    c = np.array(OMNI_MODES[names[k]][1](rng), float)
    if np.linalg.norm(c[:2]) < 0.1:
        c[:2] = 0.0
    return c


def load_model(visual=False):
    if visual:
        return mujoco.MjModel.from_xml_path(ASSET)
    spec = mujoco.MjSpec.from_file(ASSET)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for me in list(spec.meshes):
        spec.delete(me)
    return spec.compile()


class Go1Batch:
    def __init__(self, num_envs=32, seed=0, fault_prob=0.0, push_max=0.0,
                 randomize=True, obs_noise=True, command_mode="random",
                 focus_faults=None, focus_prob=0.0, visual=False, track_sigma=0.25,
                 gait_clock=False, gait_freq=2.0, phase_coef=0.5, gait_rewards=False, slip_coef=0.2):
        self.K = num_envs
        self.gait_clock = gait_clock
        self.gait_rewards = gait_rewards
        self.slip_coef = slip_coef
        self.air_threshold = 0.2 if gait_rewards else 0.4
        self.gait_freq = gait_freq
        self.phase_coef = phase_coef
        self.actor_dim = ACTOR_DIM + (2 if gait_clock else 0)
        self.critic_dim = CRITIC_DIM + (2 if gait_clock else 0)
        self.track_sigma = track_sigma
        self.focus_faults = focus_faults or []
        self.focus_prob = focus_prob
        self.rng = np.random.default_rng(seed)
        self.fault_prob = fault_prob
        self.push_max = push_max
        self.randomize = randomize
        self.pen_scale = 1.0
        self.terminate = True
        self.obs_noise = obs_noise
        self.command_mode = command_mode
        self.fixed_command = np.array([0.6, 0.0, 0.0])
        self.forced_fault = None

        base = load_model(visual)
        base.opt.timestep = SIM_DT
        self.base_model = base
        self.models = [copy.deepcopy(base) for _ in range(self.K)]
        self.datas = [mujoco.MjData(m) for m in self.models]

        self.q0 = base.key_qpos[0, 7:19].copy()
        self.ctrl_lo = base.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = base.actuator_ctrlrange[:, 1].copy()
        self.nom_kp = base.actuator_gainprm[:, 0].copy()
        self.nom_force = base.actuator_forcerange.copy()

        bid = lambda n: mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, n)
        gid = lambda n: mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.trunk_id = bid("trunk")
        self.floor_gid = gid("floor")
        self.foot_gids = np.array([gid(n) for n in ("FR", "FL", "RR", "RL")])
        self.terminal_bodies = {bid(n) for n in ("trunk", "FR_hip", "FL_hip", "RR_hip", "RL_hip")}
        self.penal_bodies = {bid(f"{l}_{p}") for l in ("FR", "FL", "RR", "RL") for p in ("thigh", "calf")}
        self.terminal_arr = np.array(sorted(self.terminal_bodies))
        self.penal_arr = np.array(sorted(self.penal_bodies))
        self.base_trunk_mass = base.body_mass[self.trunk_id]
        self.base_foot_fric = base.geom_friction[self.foot_gids, 0].copy()

        K = self.K
        self.hist = np.zeros((K, HIST, OBS_DIM), np.float32)
        self.last_action = np.zeros((K, 12), np.float32)
        self.last_last_action = np.zeros((K, 12), np.float32)
        self.last_qvel = np.zeros((K, 12))
        self.cmd = np.zeros((K, 3))
        self.step_count = np.zeros(K, np.int64)
        self.next_push = np.zeros(K, np.int64)
        self.next_cmd = np.zeros(K, np.int64)
        self.air_time = np.zeros((K, 4))
        self.last_contact = np.zeros((K, 4), bool)
        self.fault_joint = np.full(K, -1)
        self.fault_type = np.zeros(K, np.int64)
        self.fault_onset = np.zeros(K, np.int64)
        self.fault_active = np.zeros(K, bool)
        self.phase = np.zeros(K)
        self.last_foot_xy = np.zeros((K, 4, 2))
        self.frozen_target = np.zeros(K)
        self.fric = np.ones(K)
        self.dmass = np.zeros(K)
        self.kp_scale = np.ones(K)
        self.ep_stats = []

        for i in range(K):
            self._reset_env(i)

    def _restore_model(self, i):
        m = self.models[i]
        m.actuator_gainprm[:, 0] = self.nom_kp
        m.actuator_biasprm[:, 1] = -self.nom_kp
        m.actuator_forcerange[:] = self.nom_force
        m.body_mass[self.trunk_id] = self.base_trunk_mass
        m.geom_friction[self.foot_gids, 0] = self.base_foot_fric

    def _apply_randomisation(self, i):
        m = self.models[i]
        if self.randomize:
            self.fric[i] = self.rng.uniform(0.4, 1.25)
            self.dmass[i] = self.rng.uniform(-1.0, 2.0)
            self.kp_scale[i] = self.rng.uniform(0.85, 1.15)
        else:
            self.fric[i], self.dmass[i], self.kp_scale[i] = 0.8, 0.0, 1.0
        m.geom_friction[self.foot_gids, 0] = self.fric[i]
        m.body_mass[self.trunk_id] = self.base_trunk_mass + self.dmass[i]
        kp = self.nom_kp * self.kp_scale[i]
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp

    def _schedule_fault(self, i):
        self.fault_active[i] = False
        if self.forced_fault is not None:
            j, t, onset = self.forced_fault
            self.fault_joint[i], self.fault_type[i], self.fault_onset[i] = j, t, onset
        elif self.rng.random() < self.fault_prob:
            if self.focus_faults and self.rng.random() < self.focus_prob:
                self.fault_joint[i], self.fault_type[i] = self.focus_faults[self.rng.integers(len(self.focus_faults))]
            else:
                self.fault_joint[i] = self.rng.integers(12)
                self.fault_type[i] = self.rng.integers(1, len(FAULT_TYPES))
            self.fault_onset[i] = self.rng.integers(0, 200)
        else:
            self.fault_joint[i], self.fault_type[i] = -1, 0

    def _activate_fault(self, i):
        j, t, m, d = self.fault_joint[i], self.fault_type[i], self.models[i], self.datas[i]
        self.fault_active[i] = True
        if FAULT_TYPES[t] == "limp":
            m.actuator_gainprm[j, 0] = 0.0
            m.actuator_biasprm[j, 1] = 0.0
        elif FAULT_TYPES[t] == "frozen":
            self.frozen_target[i] = d.qpos[7 + j]
        elif FAULT_TYPES[t] == "weak":
            m.actuator_forcerange[j] = self.nom_force[j] * 0.25

    def _sample_command(self, i):
        if self.command_mode == "fixed":
            self.cmd[i] = self.fixed_command
        elif self.command_mode == "omni":
            self.cmd[i] = sample_omni_command(self.rng)
        else:
            c = np.array([self.rng.uniform(-0.6, 1.0), self.rng.uniform(-0.4, 0.4), self.rng.uniform(-1.0, 1.0)])
            if self.rng.random() < 0.15:
                c[:] = 0.0
            if np.linalg.norm(c[:2]) < 0.1:
                c[:2] = 0.0
            self.cmd[i] = c
        self.next_cmd[i] = self.step_count[i] + 250

    def _reset_env(self, i):
        m, d = self.models[i], self.datas[i]
        self._restore_model(i)
        self._apply_randomisation(i)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:19] = self.q0 + self.rng.uniform(-0.1, 0.1, 12)
        d.ctrl[:] = d.qpos[7:19]
        mujoco.mj_forward(m, d)
        self.last_foot_xy[i] = d.geom_xpos[self.foot_gids, :2]
        self.step_count[i] = 0
        self.last_action[i] = 0
        self.last_last_action[i] = 0
        self.last_qvel[i] = 0
        self.air_time[i] = 0
        self.last_contact[i] = True
        self.next_push[i] = self.rng.integers(150, 300)
        if self.gait_clock:
            self.phase[i] = self.rng.uniform()
        self._sample_command(i)
        self._schedule_fault(i)
        frame = self._frame(i, noise=self.obs_noise)
        self.hist[i] = frame[None]

    def _frame(self, i, noise):
        d = self.datas[i]
        R = d.xmat[self.trunk_id].reshape(3, 3)
        ang = d.qvel[3:6].copy()
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        qp = d.qpos[7:19] - self.q0
        qv = d.qvel[6:18].copy()
        if noise:
            ang = ang + self.rng.uniform(-0.2, 0.2, 3)
            grav = grav + self.rng.uniform(-0.05, 0.05, 3)
            qp = qp + self.rng.uniform(-0.01, 0.01, 12)
            qv = qv + self.rng.uniform(-1.5, 1.5, 12)
        return np.concatenate([ang * 0.25, grav, self.cmd[i] * np.array([2.0, 2.0, 0.25]),
                               qp, qv * 0.05, self.last_action[i]]).astype(np.float32)

    def _critic_obs(self, i, clean_frame, contact):
        d = self.datas[i]
        R = d.xmat[self.trunk_id].reshape(3, 3)
        linv = R.T @ d.qvel[0:3]
        fault = np.zeros(3 * 12, np.float32)
        if self.fault_active[i] and self.fault_type[i] > 0:
            fault[(self.fault_type[i] - 1) * 12 + self.fault_joint[i]] = 1.0
        return np.concatenate([clean_frame, linv * 2.0, [d.qpos[2]], fault,
                               [self.fric[i], self.dmass[i], self.kp_scale[i]],
                               contact.astype(np.float32)]).astype(np.float32)

    def _true_contact(self, d):
        g1, g2 = d.contact.geom1, d.contact.geom2
        on_floor = np.concatenate([g1[g2 == self.floor_gid], g2[g1 == self.floor_gid]]) if d.ncon else np.array([], int)
        return np.isin(self.foot_gids, on_floor)

    def _clock(self, i):
        return np.array([np.sin(2 * np.pi * self.phase[i]), np.cos(2 * np.pi * self.phase[i])], np.float32)

    def _actor_obs(self, i):
        h = self.hist[i].reshape(-1)
        return np.concatenate([self._clock(i), h]) if self.gait_clock else h

    def _critic_full(self, i, clean_frame, contact):
        c = self._critic_obs(i, clean_frame, contact)
        return np.concatenate([c, self._clock(i)]) if self.gait_clock else c

    def observe(self):
        actor = np.stack([self._actor_obs(i) for i in range(self.K)])
        critic = np.stack([self._critic_full(i, self._frame(i, False), self.last_contact[i]) for i in range(self.K)])
        return actor, critic

    def step(self, actions):
        actions = np.clip(np.asarray(actions, np.float64), -5.0, 5.0)
        K = self.K
        rew = np.zeros(K)
        done = np.zeros(K, bool)
        timeout = np.zeros(K, bool)
        actor_obs = np.zeros((K, self.actor_dim), np.float32)
        critic_obs = np.zeros((K, self.critic_dim), np.float32)
        terms = np.zeros((K, 6))

        for i in range(K):
            m, d = self.models[i], self.datas[i]
            a = actions[i]
            if self.fault_type[i] > 0 and not self.fault_active[i] and self.step_count[i] >= self.fault_onset[i]:
                self._activate_fault(i)
            target = np.clip(self.q0 + ACTION_SCALE * a, self.ctrl_lo, self.ctrl_hi)
            if self.fault_active[i] and FAULT_TYPES[self.fault_type[i]] == "frozen":
                target[self.fault_joint[i]] = self.frozen_target[i]
            d.ctrl[:] = target
            if self.push_max > 0 and self.step_count[i] >= self.next_push[i]:
                d.qvel[0:2] += self.rng.uniform(-self.push_max, self.push_max, 2)
                self.next_push[i] = self.step_count[i] + self.rng.integers(150, 300)
            for _ in range(DECIM):
                mujoco.mj_step(m, d)
            self.step_count[i] += 1
            self.phase[i] = (self.phase[i] + self.gait_freq * CTRL_DT) % 1.0
            if self.step_count[i] >= self.next_cmd[i]:
                self._sample_command(i)

            R = d.xmat[self.trunk_id].reshape(3, 3)
            linv = R.T @ d.qvel[0:3]
            angv = d.qvel[3:6]
            grav = R.T @ np.array([0.0, 0.0, -1.0])
            qv = d.qvel[6:18]
            qdd = (qv - self.last_qvel[i]) / CTRL_DT
            self.last_qvel[i] = qv
            tau = d.actuator_force
            foot_z = d.geom_xpos[self.foot_gids, 2]
            if self.gait_rewards:
                contact = self._true_contact(d)
                foot_v = (d.geom_xpos[self.foot_gids, :2] - self.last_foot_xy[i]) / CTRL_DT
                slip_w = contact.astype(float)
                if self.fault_active[i]:
                    slip_w[self.fault_joint[i] // 3] = 0.0
                slip = float(np.sum(slip_w * np.sum(foot_v ** 2, axis=1)))
            else:
                contact = foot_z < 0.032
                slip = 0.0
            self.last_foot_xy[i] = d.geom_xpos[self.foot_gids, :2]

            fell = False
            n_bad = 0
            if d.ncon:
                g1, g2 = d.contact.geom1, d.contact.geom2
                other = np.where(g1 == self.floor_gid, g2, np.where(g2 == self.floor_gid, g1, -1))
                other = other[other >= 0]
                if other.size:
                    bodies = m.geom_bodyid[other]
                    fell = bool(np.isin(bodies, self.terminal_arr).any())
                    n_bad = int((np.isin(bodies, self.penal_arr) & ~np.isin(other, self.foot_gids)).sum())
            if grav[2] > -0.3 or d.qpos[2] < 0.12:
                fell = True

            cmd = self.cmd[i]
            v_err = np.sum((cmd[:2] - linv[:2]) ** 2)
            r_lin = np.exp(-v_err / self.track_sigma)
            r_ang = np.exp(-(cmd[2] - angv[2]) ** 2 / 0.25)
            moving = np.linalg.norm(cmd[:2]) > 0.1 or abs(cmd[2]) > 0.1
            first = contact & ~self.last_contact[i]
            self.air_time[i] += CTRL_DT
            r_air = np.sum((self.air_time[i] - self.air_threshold) * first) if moving else 0.0
            self.air_time[i] *= ~contact
            self.last_contact[i] = contact
            stand = 0.0 if moving else np.sum(np.abs(d.qpos[7:19] - self.q0))
            penalty = (2.0 * linv[2] ** 2
                       + 0.05 * np.sum(angv[:2] ** 2)
                       + 1.0 * np.sum(grav[:2] ** 2)
                       + 2e-5 * np.sum(tau ** 2)
                       + 2.5e-7 * np.sum(qdd ** 2)
                       + 0.01 * np.sum((a - self.last_action[i]) ** 2)
                       + 1.0 * n_bad
                       + 0.5 * stand
                       + self.slip_coef * slip)
            r_phase = 0.0
            if self.gait_clock and moving and not self.fault_active[i]:
                stance = ((self.phase[i] + TROT_OFFSETS) % 1.0) < 0.5
                r_phase = float(np.mean(contact == stance))
            r = (1.0 * r_lin + 0.5 * r_ang + 1.0 * r_air + self.phase_coef * r_phase
                 - self.pen_scale * penalty) * CTRL_DT
            r = max(r, 0.0)
            if fell:
                r -= 1.0
            rew[i] = r
            terms[i] = [r_lin, r_ang, float(fell), np.sqrt(v_err), float(self.fault_active[i]), linv[0]]

            self.last_last_action[i] = self.last_action[i]
            self.last_action[i] = a.astype(np.float32)
            frame = self._frame(i, noise=self.obs_noise)
            self.hist[i] = np.roll(self.hist[i], -1, axis=0)
            self.hist[i, -1] = frame

            to = self.step_count[i] >= EP_LEN
            if self.terminate and (fell or to):
                self.ep_stats.append(dict(len=int(self.step_count[i]), fell=bool(fell),
                                          fault=FAULT_TYPES[self.fault_type[i]] if self.fault_active[i] else "none"))
                done[i] = True
                timeout[i] = to and not fell
                self._reset_env(i)
                frame = self.hist[i, -1]
                contact = self.last_contact[i]
            actor_obs[i] = self._actor_obs(i)
            critic_obs[i] = self._critic_full(i, self._frame(i, False), contact)
        return actor_obs, critic_obs, rew.astype(np.float32), done, timeout, terms

    def set_curriculum(self, fault_prob=None, push_max=None, pen_scale=None):
        if pen_scale is not None:
            self.pen_scale = pen_scale
        if fault_prob is not None:
            self.fault_prob = fault_prob
        if push_max is not None:
            self.push_max = push_max

    def pop_stats(self):
        s, self.ep_stats = self.ep_stats, []
        return s

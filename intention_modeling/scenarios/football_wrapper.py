import atexit
import torch
import numpy as np
from multiprocessing import Process, Pipe
from multiprocessing.shared_memory import SharedMemory
from tensordict import TensorDict
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
)

N_ACTIONS = 19
OBS_DIM = 115

_X_MIN, _X_MAX = -1.0, 1.0
_Y_MIN, _Y_MAX = -0.42, 0.42
_RENDER_W, _RENDER_H = 640, 420


def _to_px(x, y):
    px = int((x - _X_MIN) / (_X_MAX - _X_MIN) * (_RENDER_W - 1))
    py = int((y - _Y_MIN) / (_Y_MAX - _Y_MIN) * (_RENDER_H - 1))
    return px, py


def _render_obs(obs):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (_RENDER_W, _RENDER_H), (39, 125, 42))
    draw = ImageDraw.Draw(img)

    draw.rectangle([1, 1, _RENDER_W - 2, _RENDER_H - 2], outline="white", width=2)
    draw.line([_to_px(0, _Y_MIN), _to_px(0, _Y_MAX)], fill="white", width=2)
    cx, cy = _to_px(0, 0)
    r_circle = int(0.095 / (_X_MAX - _X_MIN) * _RENDER_W)
    draw.ellipse([cx - r_circle, cy - r_circle, cx + r_circle, cy + r_circle],
                 outline="white", width=2)

    goal_y = 0.044
    draw.line([_to_px(-1.0, -goal_y), _to_px(-1.0, goal_y)], fill="white", width=5)
    draw.line([_to_px( 1.0, -goal_y), _to_px( 1.0, goal_y)], fill="white", width=5)

    r = 7

    for i in range(11):
        x, y = float(obs[i * 2]), float(obs[i * 2 + 1])
        if y < -0.5:
            continue
        px, py = _to_px(x, y)
        active = obs[97 + i] > 0.5
        color = (255, 220, 0) if active else (30, 144, 255)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=1)

    for i in range(11):
        x, y = float(obs[44 + i * 2]), float(obs[44 + i * 2 + 1])
        if y < -0.5:
            continue
        px, py = _to_px(x, y)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(220, 50, 47), outline="white", width=1)

    bx, by = float(obs[88]), float(obs[89])
    px, py = _to_px(bx, by)
    rb = 5
    draw.ellipse([px - rb, py - rb, px + rb, py + rb], fill="white", outline="black", width=1)

    return np.array(img, dtype=np.uint8)


def _worker(remote, env_name, num_left, num_right, n_local_envs,
            shm_names, offset, total_envs):
    from multiprocessing.shared_memory import SharedMemory
    from gfootball.env import create_environment
    from gfootball.env import football_action_set as fas
    import numpy as np

    shm_obs  = SharedMemory(name=shm_names[0])
    shm_rew  = SharedMemory(name=shm_names[1])
    shm_done = SharedMemory(name=shm_names[2])
    shm_adv  = SharedMemory(name=shm_names[3])

    obs_buf  = np.ndarray((total_envs, num_left, OBS_DIM), dtype=np.float32, buffer=shm_obs.buf)
    rew_buf  = np.ndarray((total_envs, num_left),           dtype=np.float32, buffer=shm_rew.buf)
    done_buf = np.ndarray((total_envs,),                    dtype=np.uint8,   buffer=shm_done.buf)
    adv_buf  = np.ndarray((total_envs, num_right),          dtype=np.int64,   buffer=shm_adv.buf)

    backend_to_v1 = {a._backend_action: i for i, a in enumerate(fas.action_set_v1)}

    envs = []
    cpp_envs = []
    for _ in range(n_local_envs):
        env = create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=num_left,
            number_of_right_players_agent_controls=0,
            rewards="scoring,checkpoints",
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
        )
        envs.append(env)
        cpp_envs.append(env.unwrapped._env._env)

    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            for i, env in enumerate(envs):
                obs_buf[offset + i] = np.array(env.reset(), dtype=np.float32).reshape(num_left, OBS_DIM)
            remote.send(True)
        elif cmd == "step":
            for i, (env, cpp_env) in enumerate(zip(envs, cpp_envs)):
                obs, rew, done, _ = env.step(data[i])
                obs_arr = np.array(obs, dtype=np.float32).reshape(num_left, OBS_DIM)
                rew_arr = np.array(rew, dtype=np.float32).reshape(num_left)
                adv_arr = np.array(
                    [backend_to_v1.get(cpp_env.get_current_function_type(False, j), 0)
                     for j in range(num_right)],
                    dtype=np.int64,
                )
                if done:
                    obs_arr = np.array(env.reset(), dtype=np.float32).reshape(num_left, OBS_DIM)
                idx = offset + i
                obs_buf[idx]  = obs_arr
                rew_buf[idx]  = rew_arr
                done_buf[idx] = done
                adv_buf[idx]  = adv_arr
            remote.send(True)
        elif cmd == "close":
            for shm in (shm_obs, shm_rew, shm_done, shm_adv):
                shm.close()
            remote.close()
            break


MAX_WORKERS = 8


class FootballWrapper:
    def __init__(self, env_name, num_envs, num_left=1, num_right=1):
        self.num_envs = num_envs
        self.num_left = num_left
        self.num_right = num_right
        self._last_obs = None

        n_workers = min(num_envs, MAX_WORKERS)
        base, remainder = divmod(num_envs, n_workers)
        self._worker_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
        self._worker_offsets = [0] + list(np.cumsum(self._worker_sizes[:-1]))

        self._shm_obs  = SharedMemory(create=True, size=num_envs * num_left * OBS_DIM * 4)
        self._shm_rew  = SharedMemory(create=True, size=num_envs * num_left * 4)
        self._shm_done = SharedMemory(create=True, size=num_envs)
        self._shm_adv  = SharedMemory(create=True, size=num_envs * num_right * 8)

        self._obs_buf  = np.ndarray((num_envs, num_left, OBS_DIM), dtype=np.float32, buffer=self._shm_obs.buf)
        self._rew_buf  = np.ndarray((num_envs, num_left),           dtype=np.float32, buffer=self._shm_rew.buf)
        self._done_buf = np.ndarray((num_envs,),                    dtype=np.uint8,   buffer=self._shm_done.buf)
        self._adv_buf  = np.ndarray((num_envs, num_right),          dtype=np.int64,   buffer=self._shm_adv.buf)

        shm_names = (self._shm_obs.name, self._shm_rew.name, self._shm_done.name, self._shm_adv.name)

        self.remotes, work_remotes = zip(*[Pipe() for _ in range(n_workers)])
        self.procs = [
            Process(
                target=_worker,
                args=(wr, env_name, num_left, num_right, self._worker_sizes[i],
                      shm_names, self._worker_offsets[i], num_envs),
                daemon=True,
            )
            for i, wr in enumerate(work_remotes)
        ]
        for p in self.procs:
            p.start()
        for wr in work_remotes:
            wr.close()

        self.group_map = {"agent": num_left, "adversary": num_right}

        self.observation_spec = CompositeSpec(
            {
                "agent": CompositeSpec(
                    {"observation": UnboundedContinuousTensorSpec(shape=(OBS_DIM,))}
                ),
                "adversary": CompositeSpec(
                    {"observation": UnboundedContinuousTensorSpec(shape=(OBS_DIM,))}
                ),
            }
        )

        self.full_action_spec = CompositeSpec(
            {
                "agent": CompositeSpec(
                    {
                        "action": OneHotDiscreteTensorSpec(
                            n=N_ACTIONS, shape=(num_left, N_ACTIONS)
                        )
                    }
                ),
                "adversary": CompositeSpec(
                    {
                        "action": OneHotDiscreteTensorSpec(
                            n=N_ACTIONS, shape=(num_right, N_ACTIONS)
                        )
                    }
                ),
            }
        )
        self.full_action_spec_unbatched = self.full_action_spec

        self._closed = False
        atexit.register(self.close)

        self._adv_idx_t    = torch.zeros(num_envs, num_right, dtype=torch.long)
        self._adv_onehot_t = torch.zeros(num_envs, num_right, N_ACTIONS)

    def _reset_all(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        for remote in self.remotes:
            remote.recv()
        self._last_obs = self._obs_buf[0, 0].copy()
        return self._obs_buf

    def render(self, mode="rgb_array", **kwargs):
        if self._last_obs is None:
            return None
        return _render_obs(self._last_obs)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        for shm in (self._shm_obs, self._shm_rew, self._shm_done, self._shm_adv):
            try:
                shm.unlink()
            except Exception:
                pass
            try:
                shm.close()
            except Exception:
                pass

    def rollout(self, max_steps, policy, callback=None, **kwargs):
        num_envs = self.num_envs
        NL, NR = self.num_left, self.num_right

        self._reset_all()

        out_obs      = torch.empty(num_envs, max_steps, NL, OBS_DIM)
        out_next_obs = torch.empty(num_envs, max_steps, NL, OBS_DIM)
        out_rew      = torch.empty(num_envs, max_steps, NL, 1)
        out_done     = torch.empty(num_envs, max_steps, 1, dtype=torch.bool)
        out_adv_oh   = torch.empty(num_envs, max_steps, NR, N_ACTIONS)
        step_tds = []

        for t in range(max_steps):
            obs_t = torch.as_tensor(self._obs_buf)
            out_obs[:, t].copy_(obs_t)
            self._last_obs = self._obs_buf[0, 0].copy()
            if callback:
                callback(self, None)

            adv_obs_v = out_obs[:, t, :1].expand(num_envs, NR, OBS_DIM)
            td = TensorDict(
                source={
                    "agent": TensorDict(
                        {"observation": out_obs[:, t]},
                        batch_size=[num_envs, NL],
                    ),
                    "adversary": TensorDict(
                        {"observation": adv_obs_v},
                        batch_size=[num_envs, NR],
                    ),
                    "next": TensorDict(
                        source={
                            "agent":    TensorDict(batch_size=[num_envs, NL]),
                            "adversary": TensorDict(batch_size=[num_envs, NR]),
                        }
                    ),
                },
                batch_size=[num_envs],
            )

            policy(td)
            step_tds.append(td)

            agent_actions = td.get(("agent", "action")).argmax(-1).cpu().numpy()
            for i, remote in enumerate(self.remotes):
                start = self._worker_offsets[i]
                end   = start + self._worker_sizes[i]
                remote.send(("step", agent_actions[start:end].tolist()))

            for remote in self.remotes:
                remote.recv()

            out_next_obs[:, t].copy_(torch.as_tensor(self._obs_buf))
            out_rew[:, t, :, 0].copy_(torch.as_tensor(self._rew_buf))
            out_done[:, t, 0].copy_(torch.as_tensor(self._done_buf))

            self._adv_idx_t.copy_(torch.as_tensor(self._adv_buf))
            self._adv_onehot_t.zero_()
            self._adv_onehot_t.scatter_(2, self._adv_idx_t.unsqueeze(-1), 1.0)
            out_adv_oh[:, t].copy_(self._adv_onehot_t)

        out_done[:, -1] = True

        for td in step_tds:
            td.pop(("agent", "observation"))
            td.pop(("adversary", "observation"))

        rollout = torch.stack(step_tds, dim=1)

        out_adv_obs      = out_obs.narrow(2, 0, 1).expand(num_envs, max_steps, NR, OBS_DIM).contiguous()
        out_adv_next_obs = out_next_obs.narrow(2, 0, 1).expand(num_envs, max_steps, NR, OBS_DIM).contiguous()
        out_adv_rew      = (-out_rew.narrow(2, 0, 1)).expand(num_envs, max_steps, NR, 1).contiguous()

        rollout.set(("agent",     "observation"),              out_obs)
        rollout.set(("adversary", "observation"),              out_adv_obs)
        rollout.set(("adversary", "action"),                   out_adv_oh)
        rollout.set("done",                                    out_done)
        rollout.set("terminated",                              out_done.clone())
        rollout.set(("next", "agent",     "observation"),      out_next_obs)
        rollout.set(("next", "adversary", "observation"),      out_adv_next_obs)
        rollout.set(("next", "agent",     "reward"),           out_rew)
        rollout.set(("next", "adversary", "reward"),           out_adv_rew)
        rollout.set(("next", "done"),                          out_done.clone())
        rollout.set(("next", "terminated"),                    out_done.clone())

        return rollout

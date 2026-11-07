import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
)
import gymnasium as gym
from torchrl.envs import ParallelEnv, GymEnv
import lbforaging


class ForagingWrapper:
    def __init__(self, env_id, num_envs):
        self.num_envs = num_envs
        self.env_id = env_id

        self.envs = [gym.make(env_id) for _ in range(num_envs)]

        self.group_map = {"agent": 1, "adversary": 1}

        self.observation_spec = CompositeSpec(
            {
                "agent": CompositeSpec(
                    {"observation": UnboundedContinuousTensorSpec(shape=(21,))}
                ),
                "adversary": CompositeSpec(
                    {"observation": UnboundedContinuousTensorSpec(shape=(21,))}
                ),
            }
        )

        self.full_action_spec = CompositeSpec(
            {
                "agent": CompositeSpec(
                    {"action": OneHotDiscreteTensorSpec(n=6, shape=(1, 6))}
                ),
                "adversary": CompositeSpec(
                    {"action": OneHotDiscreteTensorSpec(n=6, shape=(1, 6))}
                ),
            }
        )
        self.full_action_spec_unbatched = self.full_action_spec

    def _reset_all(self):
        agent_obs = []
        adv_obs = []
        for env in self.envs:
            obs, _ = env.reset()
            agent_obs.append(obs[0])
            adv_obs.append(obs[1])
        agent_obs = np.stack(agent_obs, axis=0)
        adv_obs = np.stack(adv_obs, axis=0)
        return agent_obs, adv_obs

    def render(self, *args, **kwargs):
        return None

    def rollout(self, max_steps, policy, callback=None, **kwargs):
        num_envs = self.num_envs

        agent_obs, adv_obs = self._reset_all()

        rollout_steps = []

        for t in range(max_steps):
            if callback:
                callback(self, None)

            td = TensorDict(
                source={
                    "agent": TensorDict(
                        source={
                            "observation": torch.from_numpy(agent_obs).unsqueeze(1)
                        },
                        batch_size=[num_envs, 1],
                    ),
                    "adversary": TensorDict(
                        source={"observation": torch.from_numpy(adv_obs).unsqueeze(1)},
                        batch_size=[num_envs, 1],
                    ),
                    "next": TensorDict(
                        source={
                            "agent": TensorDict(batch_size=[num_envs, 1]),
                            "adversary": TensorDict(batch_size=[num_envs, 1]),
                        }
                    ),
                },
                batch_size=[num_envs],
            )

            policy(td)

            agent_action = (
                td.get(("agent", "action")).argmax(-1).squeeze(0).cpu().numpy()
            )
            adv_action = (
                td.get(("adversary", "action")).argmax(-1).squeeze(0).cpu().numpy()
            )

            next_agent_obs = np.zeros_like(agent_obs)
            next_adv_obs = np.zeros_like(adv_obs)
            rewards = np.zeros((num_envs, 2), dtype=np.float32)
            done = np.zeros((num_envs,), dtype=bool)

            for i, env in enumerate(self.envs):
                (obs_i_agent, obs_i_adv), rew_i, done_i, _, _ = env.step(
                    (int(agent_action[i]), int(adv_action[i]))
                )

                rewards[i, 0] = float(rew_i[0])
                rewards[i, 1] = float(rew_i[1])
                done[i] = done_i

                next_agent_obs[i, :] = obs_i_agent
                next_adv_obs[i, :] = obs_i_adv

            done_t = torch.from_numpy(done).unsqueeze(-1)

            td.set("done", done_t, inplace=True)
            td.set("terminated", done_t, inplace=True)
            td.set(("next", "done"), done_t, inplace=True)
            td.set(("next", "terminated"), done_t, inplace=True)

            rew_t = torch.from_numpy(rewards)
            td.set(
                ("next", "agent", "reward"),
                rew_t[:, 0].view(num_envs, 1, 1),
                inplace=True,
            )
            td.set(
                ("next", "adversary", "reward"),
                rew_t[:, 1].view(num_envs, 1, 1),
                inplace=True,
            )

            td.set(
                ("next", "agent", "observation"),
                torch.from_numpy(next_agent_obs).float().unsqueeze(1),
                inplace=True,
            )
            td.set(
                ("next", "adversary", "observation"),
                torch.from_numpy(next_adv_obs).float().unsqueeze(1),
                inplace=True,
            )

            rollout_steps.append(td)
            agent_obs, adv_obs = next_agent_obs, next_adv_obs

        rollouts = torch.stack(rollout_steps, dim=1)
        return rollouts

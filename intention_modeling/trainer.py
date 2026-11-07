import torch
import numpy as np
from tensordict.nn import TensorDictSequential
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import (
    set_exploration_type,
    ExplorationType,
)
from tqdm import tqdm

from intention_modeling.utils import DoneTransform
from intention_modeling.scenarios import CUSTOM_SCENARIOS, ForagingWrapper, KuhnWrapper, FootballWrapper


def make_env(config, num_envs, device):
    if config.scenario.name.startswith("Foraging"):
        return ForagingWrapper(config.scenario.name, num_envs)
    
    if config.scenario.name.startswith("gfootball_"):
        env_name = config.scenario.name[len("gfootball_"):]
        return FootballWrapper(
            env_name,
            num_envs,
            num_left=config.scenario.num_good_agents,
            num_right=config.scenario.num_adversaries,
        )
    
    if config.scenario.name.startswith("Kuhn"):
        return KuhnWrapper(num_envs)

    if config.scenario.name.startswith("custom"):
        scenario = CUSTOM_SCENARIOS[config.scenario.name]()
    else:
        scenario = config.scenario.name

    return VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        **vars(config.scenario),
    )


class Trainer:
    def __init__(self, config, logger, device):
        self.config = config
        self.device = device
        self.logger = logger

        self.env = make_env(
            config, config.n_steps_per_iter // config.scenario.max_steps, device
        )
        self.done_transform = DoneTransform(self.env)
        self.test_env = make_env(config, config.num_test_envs, device)

    def train(self, main_agent, opponents, intention_model, pretrained_opponents=False):
        pbar = tqdm(total=self.config.n_iters, desc="iteration = 0")
        total_env_steps = 0

        explore_policy = TensorDictSequential(
            intention_model.process,
            main_agent.explore_policy,
            opponents.explore_policy,
            intention_model.postprocess,
        )

        for iter in range(self.config.n_iters):
            if pretrained_opponents:
                opponents.reset_policies(iter)
            intention_model.reset(
                self.config.n_steps_per_iter // self.config.scenario.max_steps
            )

            with torch.no_grad():
                with set_exploration_type(ExplorationType.RANDOM):
                    rollouts = self.env.rollout(
                        max_steps=self.config.scenario.max_steps,
                        policy=explore_policy,
                        auto_cast_to_device=False,
                        break_when_any_done=False,
                        auto_reset=True,
                    )

            total_env_steps += int(np.prod(rollouts.batch_size))
            rollouts = self.done_transform(rollouts)
            main_agent.handle_next(rollouts)

            logs = intention_model.update(rollouts)
            logs |= main_agent.update(rollouts)
            if not pretrained_opponents:
                logs |= opponents.update(rollouts)

            if iter % self.config.evaluate_every == 0:
                self.eval(
                    main_agent, opponents, intention_model, iter, total_env_steps, logs, pretrained_opponents
                )
            pbar.set_description(f"iteration = {iter}", refresh=False)
            pbar.update()

    def eval(self, main_agent, opponents, intention_model, iter, total_env_steps, logs, pretrained_opponents):
        policy = TensorDictSequential(
            intention_model.process, main_agent.policy, opponents.policy, intention_model.postprocess,
        )
        intention_model.reset(self.config.num_test_envs)

        with torch.no_grad():
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                rollouts = self.test_env.rollout(
                    max_steps=self.config.scenario.max_steps,
                    policy=policy,
                    auto_cast_to_device=False,
                    break_when_any_done=False,
                    auto_reset=True,
                )
        logs |= intention_model.eval(rollouts)
        if pretrained_opponents:
            logs |= opponents.eval()
        if main_agent.embedding_mixer and hasattr(main_agent.embedding_mixer.module, "eval_"):
            logs |= main_agent.embedding_mixer.module.eval_(rollouts.select(*main_agent.embedding_mixer.in_keys))

        rollouts = list(rollouts.unbind(0))
        for k, r in enumerate(rollouts):
            next_done = r.get(("next", "done")).sum(
                tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
                dtype=torch.bool,
            )
            done_index = next_done.nonzero(as_tuple=True)[0][0]
            rollouts[k] = r[: done_index + 1]

        for group, agents in self.env.group_map.items():
            rewards = [
                td.get(("next", group, "reward")).sum(0).mean() for td in rollouts
            ]
            logs[f"eval/{group}_episode_reward_min"] = min(rewards)
            logs[f"eval/{group}_episode_reward_max"] = max(rewards)
            logs[f"eval/{group}_episode_reward_mean"] = sum(rewards) / len(rollouts)

        logs["eval/episode_len_mean"] = sum(
            [td.batch_size[0] for td in rollouts]
        ) / len(rollouts)
        logs["train/training_iteration"] = iter
        logs["train/total_env_steps"] = total_env_steps
        self.logger.log(logs)

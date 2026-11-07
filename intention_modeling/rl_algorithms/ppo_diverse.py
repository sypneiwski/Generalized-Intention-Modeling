import torch
import torch.nn as nn
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, OneHotCategorical
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule, TensorDictSequential

from intention_modeling.utils import load_all_models


class PPO_Diverse:
    def __init__(
        self,
        config,
        n_agents,
        group,
        obs_dim,
        obs_key,
        act_shape,
        diversity_alpha,
        device,
        opponent_population=[],
    ):
        self.config = config
        self.device = device
        self.group = group
        self.n_agents = n_agents
        self.opponent_population = opponent_population
        self.diversity_alpha = diversity_alpha
        print(f"Diversity alpha set to {self.diversity_alpha}!")

        actor_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=act_shape,
            n_agents=n_agents,
            centralised=self.config.centralised.actor,
            share_params=self.config.share_parameters.actor,
            device=self.device,
            depth=3,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ).to(self.device)

        actor_module = TensorDictModule(
            actor_net, in_keys=obs_key, out_keys=[(group, "logits")]
        )

        self.policy = ProbabilisticActor(
            module=actor_module,
            in_keys=[(group, "logits")],
            out_keys=[(group, "action")],
            distribution_class=OneHotCategorical,
            return_log_prob=True,
            log_prob_key=(group, "sample_log_prob"),
        )
        self.explore_policy = self.policy

        critic_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralised=self.config.centralised.critic,
            share_params=self.config.share_parameters.critic,
            device=device,
            depth=3,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ).to(self.device)

        self.critic = TensorDictModule(
            module=critic_net, in_keys=obs_key, out_keys=[(group, "state_value")]
        )

        self.loss = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coeff=self.config.entropy_coefficient,
        )
        self.loss.set_keys(
            value=(group, "state_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            sample_log_prob=(group, "sample_log_prob"),
            action=(group, "action"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
        )
        self.loss.make_value_estimator(
            ValueEstimators.GAE, gamma=self.config.gamma, lmbda=self.config.lmbda
        )

        self.optimizer = torch.optim.Adam(
            self.loss.parameters(), lr=self.config.learning_rate
        )

    def update(self, rollouts):
        with torch.no_grad():
            self.loss.value_estimator(
                rollouts,
                params=self.loss.critic_network_params,
                target_params=self.loss.target_critic_network_params,
            )

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(rollouts.batch_size[0], device=self.device),
            sampler=SamplerWithoutReplacement(shuffle=False),
            batch_size=self.config.minibatch_size,
        )
        replay_buffer.extend(rollouts)

        total_loss = 0.0
        num_updates = 0
        for epoch in range(self.config.n_epochs):
            for _ in range(rollouts.batch_size[0] // self.config.minibatch_size):
                minibatch = replay_buffer.sample()
                self.optimizer.zero_grad()

                loss = self.loss(minibatch)
                diversity_loss = (
                    self.compute_surprise_loss(minibatch) * self.diversity_alpha
                    if self.diversity_alpha != 0
                    and self.n_agents + len(self.opponent_population) > 1
                    else 0
                )

                loss_values = (
                    loss["loss_objective"]
                    + loss["loss_critic"]
                    + loss["loss_entropy"]
                    - diversity_loss
                )
                loss_values.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.loss.parameters(), self.config.max_grad_norm
                )

                self.optimizer.step()
                total_loss += loss_values.item()
                num_updates += 1

        return {
            f"train/{self.group}_ppo_loss": total_loss / num_updates,
            f"train/{self.group}_diversity_loss": diversity_loss,
        }

    def compute_surprise_loss(self, minibatch):
        obs_all = minibatch.get((self.group, "observation"))
        n_learn = obs_all.shape[2]
        kl_terms = []

        for i in range(n_learn):
            obs_i = obs_all[:, :, i : i + 1, :]
            obs_i_tiled = obs_i.expand(-1, -1, n_learn, -1)

            td_i = minibatch.clone()
            td_i.set((self.group, "observation"), obs_i_tiled)

            dist = self.policy.get_dist(td_i)
            logits_all = dist.logits
            probs_all = torch.softmax(logits_all, dim=-1)

            p_i = probs_all[:, :, i, :]
            other_probs = []

            for j in range(n_learn):
                if j == i:
                    continue
                other_probs.append(probs_all[:, :, j : j + 1, :].detach())

            for opp in self.opponent_population:
                with torch.no_grad():
                    dist_opp = opp.get_dist(td_i)
                    p_opp = torch.softmax(dist_opp.logits, dim=-1)
                for p_opp_i in p_opp.unbind(-2):
                    other_probs.append(p_opp_i.unsqueeze(-2))

            other_probs = torch.cat(other_probs, dim=2).mean(dim=2)

            kl_i = (
                p_i * (p_i.clamp_min(1e-8).log() - other_probs.clamp_min(1e-8).log())
            ).sum(
                dim=-1
            )
            kl_terms.append(kl_i)

        kl_all = torch.stack(kl_terms, dim=2)
        return kl_all.mean()

    def handle_next(self, rollouts):
        pass

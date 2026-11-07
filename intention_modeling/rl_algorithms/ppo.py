import torch
import torch.nn as nn
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, OneHotCategorical
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule, TensorDictSequential


class PPO:
    def __init__(
        self,
        config,
        n_agents,
        group,
        obs_dim,
        obs_key,
        act_spec,
        device,
        embedding_mixer=None,
    ):
        self.config = config
        self.device = device
        self.group = group
        self.embedding_mixer = embedding_mixer

        actor_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=act_spec.shape[-1],
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
        safe_logits = TensorDictModule(
            lambda t: torch.nan_to_num(t, nan=0.0, posinf=20.0, neginf=-20.0),
            in_keys=[(group, "logits")],
            out_keys=[(group, "logits")],
        )
        actor_module = TensorDictSequential(actor_module, safe_logits)
        if embedding_mixer is not None:
            actor_module = TensorDictSequential(embedding_mixer, actor_module)

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
        num_skipped = 0
        for _ in range(self.config.n_epochs):
            for _ in range(rollouts.batch_size[0] // self.config.minibatch_size):
                minibatch = replay_buffer.sample()
                self.optimizer.zero_grad()

                loss = self.loss(minibatch)
                loss_values = (
                    loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                )

                if not torch.isfinite(loss_values):
                    num_skipped += 1
                    print(f"[PPO:{self.group}] skipping non-finite loss: {loss_values.item()}")
                    continue

                loss_values.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.loss.parameters(), self.config.max_grad_norm
                )

                self.optimizer.step()
                total_loss += loss_values.item()
                num_updates += 1

        denom = num_updates or 1
        logs = {
            f"train/{self.group}_ppo_loss": total_loss / denom,
            f"train/{self.group}_ppo_skipped_updates": num_skipped,
        }
        return logs

    def handle_next(self, rollouts):
        if self.embedding_mixer is None:
            return

        embeddings = rollouts.get(*self.embedding_mixer.out_keys)
        next_embeddings = torch.zeros_like(embeddings)
        next_embeddings[:, :-1] = embeddings[:, 1:]
        rollouts.set(("next", *self.embedding_mixer.out_keys), next_embeddings)

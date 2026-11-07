# Implementation of the OMG algorithm from the paper
# Opponent Modeling based on Subgoal Inference, Yu et al., NeurIPS 2024
# adapted from: https://openreview.net/forum?id=Lt6wO0oZ8k

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abc import abstractmethod
from functools import reduce
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from intention_modeling.utils import load_all_models


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(RNN, self).__init__()
        self.f = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, inputs, hidden_state):
        x_shape = inputs.shape
        x = inputs.reshape(-1, x_shape[-1])
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h_out = self.f(x, h_in)
        return h_out.view(x_shape[:-1] + torch.Size([-1])), h_out.detach()

    def init_hidden(self):
        return self.f.weight_hh.new(1, self.hidden_dim).zero_()


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def _encode(self, input):
        raise NotImplementedError

    def _decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size, current_device, **kwargs):
        raise NotImplementedError

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass


class VanillaVAE(BaseVAE):
    def __init__(self, input_dim, latent_dim, hidden_dims, **kwargs):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []

        i_d = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(i_d, h_dim), nn.LeakyReLU()))
            i_d = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.decoder_output = nn.Linear(hidden_dims[-1], input_dim)

    def forward(self, inputs):
        mu, log_var = self._encode(inputs)
        z = self._reparameterize(mu, log_var)
        return z

    def loss_func(self, inputs):
        mu, log_var = self._encode(inputs)
        z = self._reparameterize(mu, log_var)
        recons = self._decode(z)
        recon_loss = F.mse_loss(inputs, recons)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1, keepdim=True)
        )
        return recon_loss, kld_loss

    def generate(self, inputs):
        mu, log_var = self._encode(inputs)
        z = self._reparameterize(mu, log_var)
        recons = self._decode(z)
        return [inputs, recons, mu, log_var, z]

    def _encode(self, inputs):
        result = self.encoder(inputs)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def _decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.decoder_output(result)
        return result

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class ConditionalVAE(BaseVAE):
    def __init__(
        self, input_dim, conditioned_input_dim, latent_dim, hidden_dims, **kwargs
    ):
        super(ConditionalVAE, self).__init__()

        self.input_dim = input_dim
        self.condition_input_dim = conditioned_input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        modules = []

        i_d = input_dim + conditioned_input_dim
        for h_dim in hidden_dims[:-1]:
            modules.append(nn.Sequential(nn.Linear(i_d, h_dim), nn.LeakyReLU()))
            i_d = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(
            latent_dim + conditioned_input_dim, hidden_dims[-1]
        )
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.decoder_output = nn.Linear(hidden_dims[-1], input_dim)

    def forward(self, inputs, cond_inputs):
        x = torch.cat([inputs, cond_inputs], dim=-1)
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        return z

    def forward_full(self, inputs, cond_inputs):
        x = torch.cat([inputs, cond_inputs], dim=-1)
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        return z, mu, log_var

    def _encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def _decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.decoder_output(result)
        return result

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def generate(self, inputs, cond_inputs):
        x = torch.cat([inputs, cond_inputs], dim=-1)
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)

        x = torch.cat([z, cond_inputs], dim=-1)
        recons = self._decode(x)
        return [inputs, recons, mu, log_var, z]


def si_subgoals(tensordict, critic, vae, obs_for_vae, config):
    batch_size, max_steps = tensordict.shape
    horizon = config.horizon

    with torch.no_grad():
        obs_for_eval = tensordict.get(("agent", "observation"))[:, :-1]
        obs_for_vae = obs_for_vae[:, :-1].reshape(-1, obs_for_vae.shape[-1])

        _, _, mu, log_var, z = vae.generate(obs_for_vae)
        subgoal = z.view(batch_size, max_steps - 1, 1, -1)
        mu = mu.view(batch_size, max_steps - 1, 1, -1)
        log_var = log_var.view(batch_size, max_steps - 1, 1, -1)

        obs_for_eval = (
            obs_for_eval.transpose(1, 2).unsqueeze(3).repeat(1, 1, 1, horizon, 1)
        )
        temp_subgoal = subgoal.transpose(1, 2)
        subgoal_for_eval = [temp_subgoal.unsqueeze(3)]
        for h in range(1, horizon):
            temp = torch.cat(
                [
                    temp_subgoal[:, :, h:, :],
                    temp_subgoal[:, :, -1:, :].repeat(1, 1, h, 1),
                ],
                dim=2,
            ).unsqueeze(3)
            subgoal_for_eval.append(temp)
        subgoal_for_eval = torch.cat(subgoal_for_eval, dim=3)

        eval_inputs = torch.cat([obs_for_eval, subgoal_for_eval], dim=-1)
        eval_inputs = eval_inputs.transpose(1, 2)
        eval_inputs = eval_inputs.transpose(2, 3)

        value = critic.module(eval_inputs).mean(dim=-1)

        if config.subgoal_mode == "max":
            k = torch.argmax(value, dim=2)
        else:
            k = torch.argmin(value, dim=2)
        base_idx = (
            torch.arange(max_steps - 1)
            .expand(batch_size, -1)
            .view(batch_size, 1, -1)
            .transpose(1, 2)
            .to(k.device)
        )
        k = (
            torch.min(base_idx + k, torch.LongTensor([max_steps - 2]).to(k.device))
            .unsqueeze(-1)
            .repeat(1, 1, 1, config.embedding_dim)
        )

        final_subgoal = torch.gather(subgoal, 1, k)
        final_mu = torch.gather(mu, 1, k)
        final_log_var = torch.gather(log_var, 1, k)
    return final_subgoal, final_mu, final_log_var


class OMGIM:
    def __init__(
        self,
        critic,
        n_iters,
        obs_dim,
        act_dim,
        n_modelled_agents,
        modelled_obs_dim,
        config,
        device,
        vae=None,
    ):
        self.critic = critic
        self._obs_dim = obs_dim
        self._modelled_obs_dim = modelled_obs_dim
        self._n_modelled_agents = n_modelled_agents
        self._act_dim = act_dim
        self._hidden_dim = 128
        self._subgoal_dim = config.embedding_dim
        self._modelled_dim = obs_dim + modelled_obs_dim * n_modelled_agents
        self._eta = 0.5
        self._eta_delta = 0.5 / n_iters

        self.fc = nn.Linear(obs_dim, self._hidden_dim // 4).to(device)
        self.cond_rnn = RNN(obs_dim + act_dim, self._hidden_dim // 4 * 3).to(device)
        self.cvae = ConditionalVAE(
            input_dim=self._hidden_dim // 4,
            conditioned_input_dim=self._hidden_dim // 4 * 3,
            latent_dim=self._subgoal_dim,
            hidden_dims=[128, 128],
        ).to(device)
        self.vae = VanillaVAE(
            input_dim=self._modelled_dim,
            latent_dim=self._subgoal_dim,
            hidden_dims=[128, 128],
        ).to(device)

        self.device = device
        self.config = config
        self.training_vae = vae is None

        self.vae_optimizer = torch.optim.Adam(
            self.vae_parameters(), lr=self.config.lr_vae
        )
        self.omg_optimizer = torch.optim.Adam(
            self.omg_parameters(), lr=self.config.lr_omg
        )

        print(f"Training VAE set to: {self.training_vae}")
        if not self.training_vae:
            loaded = load_all_models(vae, device)
            assert len(loaded) == 1
            self.vae = loaded[0].to(device)

        self.process = TensorDictModule(
            self.forward,
            in_keys=[("agent", "observation")],
            out_keys=[
                ("agent", "omg"),
                ("agent", "infer_mu"),
                ("agent", "infer_log_var"),
            ]
            if not self.training_vae
            else [("agent", "observation")],
        )

        self.postprocess = TensorDictModule(
            self._postprocess,
            in_keys=[("agent", "action"), ("agent", "observation")],
            out_keys=[("agent", "action")],
        )

    def _postprocess(self, previous_action, previous_obs):
        self.prev_act = previous_action
        self.prev_obs = previous_obs
        return previous_action

    def _preprocess(self, tensordict):
        prev_action = torch.zeros_like(tensordict.get(("agent", "action")))
        prev_action[:, 1:] = tensordict.get(("agent", "action"))[:, :-1]
        tensordict.set(("agent", "prev_act"), prev_action, inplace=True)

        prev_obs = torch.zeros_like(tensordict.get(("agent", "observation")))
        prev_obs[:, 1:] = tensordict.get(("agent", "observation"))[:, :-1]
        tensordict.set(("agent", "prev_obs"), prev_obs, inplace=True)

    def get_vae_input(self, td):
        B, T = td.shape
        return torch.cat(
            (
                td.get(("agent", "observation")),
                td.get(("adversary", "observation")).reshape(B, T, 1, -1),
            ),
            dim=-1,
        )

    def forward(self, obs):
        if self.training_vae:
            return obs

        cond_inputs = torch.cat((self.prev_obs, self.prev_act), dim=-1)
        old_shape = obs.shape[:-1]
        num = reduce((lambda x, y: x * y), list(obs.shape[:-1]))

        x = F.relu(self.fc(obs)).view(num, -1)
        cond_x, self.hidden_state = self.cond_rnn(cond_inputs, self.hidden_state)
        out, mu, log_var = self.cvae.forward_full(x, cond_x.view(num, -1))
        out = out.view(old_shape + torch.Size([-1]))

        return out, mu.unsqueeze(1), log_var.unsqueeze(1)

    def loss_omg(self, tensordict):
        batch_size, max_steps = tensordict.shape

        cvae_output = []
        cvae_mu = []
        cvae_log_var = []
        cvae_input = []
        infer_mu = tensordict.get(("agent", "infer_mu"))[:, :-1]
        infer_log_var = tensordict.get(("agent", "infer_log_var"))[:, :-1]

        self.reset(batch_size)
        for t in range(max_steps - 1):
            fc_inputs = tensordict.get(("agent", "observation"))[:, t]
            cond_inputs = torch.cat(
                (
                    tensordict.get(("agent", "prev_obs"))[:, t],
                    tensordict.get(("agent", "prev_act"))[:, t],
                ),
                dim=-1,
            )

            x = F.relu(self.fc(fc_inputs))
            cond_x, self.hidden_state = self.cond_rnn(cond_inputs, self.hidden_state)
            x, recons, mu, log_var, _ = self.cvae.generate(
                x.view(batch_size, -1), cond_x.view(batch_size, -1)
            )

            cvae_input.append(x.view(batch_size, 1, 1, -1))
            cvae_output.append(recons.view(batch_size, 1, 1, -1))
            cvae_mu.append(mu.view(batch_size, 1, 1, -1))
            cvae_log_var.append(log_var.view(batch_size, 1, 1, -1))

        obs_for_vae = self.get_vae_input(tensordict)
        cvae_input = torch.cat(cvae_input, dim=1)
        cvae_output = torch.cat(cvae_output, dim=1)
        cvae_mu = torch.cat(cvae_mu, dim=1)
        cvae_log_var = torch.cat(cvae_log_var, dim=1)

        subgoal, vae_mu, vae_log_var = si_subgoals(
            tensordict, self.critic, self.vae, obs_for_vae, self.config
        )
        vae_mu = vae_mu.reshape(*cvae_mu.shape)
        vae_log_var = vae_log_var.reshape(*cvae_mu.shape)

        recons_loss = F.mse_loss(cvae_output, cvae_input, reduction="none").mean(dim=-1)

        if self._eta < np.random.random():
            omg_loss = -0.5 * torch.sum(
                1
                + cvae_log_var
                - vae_log_var.detach()
                - ((vae_mu.detach() - cvae_mu) ** 2 + cvae_log_var.exp())
                / vae_log_var.detach().exp(),
                dim=-1,
            )
        else:
            omg_loss = -0.5 * torch.sum(
                1
                + cvae_log_var
                - infer_log_var.detach()
                - ((infer_mu.detach() - cvae_mu) ** 2 + cvae_log_var.exp())
                / infer_log_var.detach().exp(),
                dim=-1,
            )
        return recons_loss.mean(), omg_loss.mean()

    def update_omg(self, tensordict):
        recons_loss, omg_loss = self.loss_omg(tensordict)

        self.omg_optimizer.zero_grad()
        loss = recons_loss + self.config.omg_cvae_alpha * omg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.omg_parameters(), self.config.max_grad_norm)

        self.omg_optimizer.step()
        return recons_loss.detach().item(), omg_loss.detach().item()

    def update(self, rollouts):
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(rollouts.batch_size[0], device=self.device),
            sampler=SamplerWithoutReplacement(shuffle=False),
            batch_size=self.config.minibatch_size,
        )
        self._preprocess(rollouts)
        replay_buffer.extend(rollouts)

        update_func = self.update_vae if self.training_vae else self.update_omg

        total_first_loss = 0.0
        total_second_loss = 0.0
        second_loss_name = (
            "train/train_vae_loss" if self.training_vae else "train/train_omg_loss"
        )
        num_updates = 0

        for _ in range(self.config.n_epochs):
            for _ in range(rollouts.batch_size[0] // self.config.minibatch_size):
                num_updates += 1
                minibatch = replay_buffer.sample()

                first_loss, second_loss = update_func(minibatch)

                total_first_loss += first_loss
                total_second_loss += second_loss

        self._eta += self._eta_delta
        return {
            "train/train_recon_loss": total_first_loss / num_updates,
            second_loss_name: total_second_loss / num_updates,
            "train/eta": self._eta,
        }

    def update_vae(self, tensordict):
        assert len(tensordict.shape) == 2

        obs_for_vae = self.get_vae_input(tensordict).reshape(-1, self._modelled_dim)
        recons_loss, kld_loss = self.vae.loss_func(obs_for_vae)

        self.vae_optimizer.zero_grad()
        loss = recons_loss + self.config.omg_vae_alpha * kld_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vae_parameters(), self.config.max_grad_norm)

        self.vae_optimizer.step()
        return recons_loss.detach().item(), kld_loss.detach().item()

    def vae_parameters(self):
        return self.vae.parameters()

    def omg_parameters(self):
        return (
            list(self.cvae.parameters())
            + list(self.cond_rnn.parameters())
            + list(self.fc.parameters())
        )

    def reset(self, batch_dim):
        self.hidden_state = self.cond_rnn.init_hidden().expand(batch_dim, -1)
        self.prev_obs = torch.zeros(batch_dim, 1, self._obs_dim, device=self.device)
        self.prev_act = torch.zeros(batch_dim, 1, self._act_dim, device=self.device)

    def eval(self, rollouts):
        logs = {}
        with torch.no_grad():
            if not self.training_vae:
                self._preprocess(rollouts)
                recons_loss, second_loss = self.loss_omg(rollouts)

                logs["eval/omg_recon_loss"] = recons_loss.detach().item()
                logs["eval/omg_loss"] = second_loss.detach().item()
        return logs

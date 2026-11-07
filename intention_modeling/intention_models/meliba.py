# Implementation of the MeLIBA algorithm from the paper
# Deep Interactive Bayesian Reinforcement Learning via Meta-Learning, Zintgraf et al., AAMAS 2021

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule


class MeLIBAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim_m, latent_dim_mt):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc_m_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mt_hidden = nn.Linear(hidden_dim + latent_dim_m, hidden_dim)

        self.fc_m_mu = nn.Linear(hidden_dim, latent_dim_m)
        self.fc_m_logvar = nn.Linear(hidden_dim, latent_dim_m)
        self.fc_mt_mu = nn.Linear(hidden_dim, latent_dim_mt)
        self.fc_mt_logvar = nn.Linear(hidden_dim, latent_dim_mt)

    def forward(self, x, h_in):
        x = F.relu(self.input_fc(x))
        output, h_out = self.lstm(x, h_in)

        m_hidden = F.relu(self.fc_m_hidden(output))
        m_mu = torch.clamp(self.fc_m_mu(m_hidden), min=-10.0, max=10.0)
        m_logvar = torch.clamp(self.fc_m_logvar(m_hidden), min=-10.0, max=10.0)

        m_sample = self._reparameterize(m_mu, m_logvar)
        mt_input = torch.cat([output, m_sample], dim=-1)

        mt_hidden = F.relu(self.fc_mt_hidden(mt_input))
        mt_mu = torch.clamp(self.fc_mt_mu(mt_hidden), min=-10.0, max=10.0)
        mt_logvar = torch.clamp(self.fc_mt_logvar(mt_hidden), min=-10.0, max=10.0)

        return m_mu, m_logvar, mt_mu, mt_logvar, h_out

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class MeLIBADecoder(nn.Module):
    def __init__(self, ld_m, ld_mt, all_obs_dim, act_dim, n_opps):
        super().__init__()

        self.decoder_m_embed = nn.Linear(ld_m, 32)
        self.decoder_mt_embed = nn.Linear(ld_mt + 32, 64)
        self.decoder_joint = nn.Linear(32 + all_obs_dim, 64)
        self.decoder_gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.decoder_out_hidden = nn.Linear(64, 32)
        self.decoder_out = nn.Linear(32, n_opps * act_dim)

    def forward(self, z_m, z_mt, joint_obs_sequence):
        B, H, _ = joint_obs_sequence.shape

        hm = F.relu(self.decoder_m_embed(z_m))
        mt_input = torch.cat([z_mt, hm], dim=-1)
        hmt = F.relu(self.decoder_mt_embed(mt_input))

        hm_expanded = hm.unsqueeze(1).expand(-1, H, -1)
        joint_input = torch.cat([hm_expanded, joint_obs_sequence], dim=-1)
        gru_input = F.relu(self.decoder_joint(joint_input))

        h_0 = hmt.unsqueeze(0).contiguous()
        gru_out, _ = self.decoder_gru(gru_input, h_0)
        out_hidden = F.relu(self.decoder_out_hidden(gru_out))
        pred_actions = self.decoder_out(out_hidden)

        return pred_actions


class MeLIBA_IM:
    def __init__(
        self, config, own_obs_dim, opp_obs_dim, act_dim, n_opps, device, hidden_dim=128
    ):
        self.config = config
        self.device = device
        self.n_opps = n_opps
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.ld_m = config.m_dim
        self.ld_mt = config.mt_dim

        self.encoder = MeLIBAEncoder(
            own_obs_dim + act_dim, self.hidden_dim, self.ld_m, self.ld_mt
        ).to(device)
        self.decoder = MeLIBADecoder(
            self.ld_m, self.ld_mt, own_obs_dim + opp_obs_dim * n_opps, act_dim, n_opps
        )
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.lr_vae,
        )

        self.process = TensorDictModule(
            self.forward,
            in_keys=[("agent", "observation")],
            out_keys=[("agent", "meliba")],
        )
        self.postprocess = TensorDictModule(
            self._postprocess,
            in_keys=[("agent", "action")],
            out_keys=[("agent", "action")],
        )

    def forward(self, observations):
        encoder_input = torch.cat((observations, self.prev_actions), dim=-1)
        m_mu, m_lv, mt_mu, mt_lv, self.h_state = self.encoder(
            encoder_input, self.h_state
        )
        belief = torch.cat([m_mu, m_lv, mt_mu, mt_lv], dim=-1)
        return belief

    def _preprocess(self, rollouts):
        prev_action = torch.zeros_like(rollouts.get(("agent", "action")))
        prev_action[:, 1:] = rollouts.get(("agent", "action"))[:, :-1]
        rollouts.set(("agent", "prev_act"), prev_action, inplace=True)

    def _postprocess(self, actions):
        self.prev_actions = actions
        return actions

    def compute_loss(self, minibatch):
        own_obs = minibatch.get(("agent", "observation")).squeeze(-2)
        B, T, _ = own_obs.shape
        opp_obs = minibatch.get(("adversary", "observation")).reshape(B, T, -1)
        joint_obs = torch.cat([own_obs, opp_obs], dim=-1)
        hist_input = torch.cat(
            [own_obs, minibatch.get(("agent", "prev_act")).squeeze(-2)], dim=-1
        )
        adv_acts = minibatch.get(("adversary", "action"))

        h_enc = (
            torch.zeros((1, B, self.hidden_dim)).to(self.device),
            torch.zeros((1, B, self.hidden_dim)).to(self.device),
        )
        m_mu, m_lv, mt_mu, mt_lv, _ = self.encoder(hist_input, h_enc)

        mu_full = torch.cat([m_mu, mt_mu], dim=-1)
        lv_full = torch.cat([m_lv, mt_lv], dim=-1)

        prior_mu = torch.zeros_like(mu_full)
        prior_mu[:, 1:] = mu_full[:, :-1].detach()
        prior_lv = torch.zeros_like(lv_full)
        prior_lv[:, 1:] = lv_full[:, :-1].detach()

        kl_loss = self._kl_divergence(mu_full, lv_full, prior_mu, prior_lv)

        z_m = self._reparameterize(m_mu, m_lv)
        z_mt = self._reparameterize(mt_mu, mt_lv)

        total_recon_loss = 0.0
        for t in range(T):
            curr_z_m = z_m[:, t]
            curr_z_mt = z_mt[:, t]

            future_joint_obs = joint_obs[:, t:]
            future_targets = adv_acts[:, t:]

            preds = self.decoder(curr_z_m, curr_z_mt, future_joint_obs)
            probs = F.softmax(preds.view(*future_targets.shape), dim=-1)
            step_loss = (
                -torch.log((future_targets * probs).sum(dim=-1)).sum(dim=-1).mean()
            )
            total_recon_loss += step_loss

        avg_recon_loss = total_recon_loss / T
        return avg_recon_loss + self.config.kl_weight * kl_loss

    def reset(self, batch_dim):
        self.h_state = (
            torch.zeros((1, batch_dim, self.hidden_dim)).to(self.device),
            torch.zeros((1, batch_dim, self.hidden_dim)).to(self.device),
        )
        self.prev_actions = torch.zeros((batch_dim, 1, self.act_dim)).to(self.device)

    def update(self, rollouts):
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(rollouts.batch_size[0], device=self.device),
            sampler=SamplerWithoutReplacement(shuffle=False),
            batch_size=self.config.minibatch_size,
        )
        self._preprocess(rollouts)
        replay_buffer.extend(rollouts)

        total_loss = 0.0
        num_updates = 0
        for _ in range(self.config.n_epochs):
            for _ in range(rollouts.batch_size[0] // self.config.minibatch_size):
                num_updates += 1
                minibatch = replay_buffer.sample()
                loss = self.compute_loss(minibatch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                total_loss += loss.item()

        return {"train/meliba_vae_loss": total_loss / num_updates}

    def _kl_divergence(self, mu1, lv1, mu2, lv2):
        kl = 0.5 * (
            lv2 - lv1 + (torch.exp(lv1) + (mu1 - mu2) ** 2) / torch.exp(lv2) - 1
        )
        return kl.sum(dim=-1).mean()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def eval(self, rollouts):
        self._preprocess(rollouts)
        with torch.no_grad():
            loss = self.compute_loss(rollouts)
        return {f"eval/meliba_vae_loss": loss.item()}

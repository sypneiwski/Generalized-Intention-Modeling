import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictSequential
from torchrl.envs import set_exploration_type, ExplorationType

from intention_modeling.utils import load_config, MODEL_DIR
from intention_modeling.trainer import make_env
from intention_modeling.rl_algorithms.ppo import PPO
from intention_modeling.opponents_handler import get_opponent_policies
from intention_modeling.mixers import MIXERS
from intention_modeling.intention_models.encoder_decoder import (
    EncoderDecoderIM,
    EncoderDecoderList,
)


Z_MIX_KEY = ("agent", "embedding")
Z_REWARD_KEY = ("agent", "RTG")


KUHN_SEEN_CLASSES = 6


def setup_opponent_labels(opponents):
    if hasattr(opponents, "test_set_policies"):
        n = len(opponents.test_set_policies)
        if n < 2:
            raise ValueError(f"Need >= 2 pooled opponents to probe, got {n}.")
        return "test_idx", n
    if hasattr(opponents, "test_heuristic_id"):
        opponents.test_on_unseen = False
        return "test_heuristic_id", KUHN_SEEN_CLASSES
    raise ValueError(
        "Opponent identity probe needs an opponent pool (POPP / LBF / Kuhn); "
        "this scenario has none."
    )


def build_and_load(config, checkpoint, device, num_envs, random_encoder=False):
    env = make_env(config.trainer, num_envs, device)

    own_obs_dim = env.observation_spec["agent", "observation"].shape[-1]
    opp_obs_dim = env.observation_spec["adversary", "observation"].shape[-1]
    act_dim = env.full_action_spec["agent", "action"].shape[-1]
    n_opps = config.trainer.scenario.num_adversaries

    intention_models = []
    for name in checkpoint["meta"]["intention_models"]:
        im = EncoderDecoderIM(
            model=name,
            config=config.intention_modeling,
            own_obs_dim=own_obs_dim,
            opp_obs_dim=opp_obs_dim,
            act_dim=act_dim,
            n_opps=n_opps,
            device=device,
        )
        if not random_encoder:
            im.encoder.load_state_dict(checkpoint["encoders"][name])
        im.encoder.eval()
        for p in im.encoder.parameters():
            p.requires_grad_(False)
        intention_models.append(im)
    intention_model = EncoderDecoderList(intention_models)

    embedding_keys = []
    for im in intention_models:
        embedding_keys += im.process.out_keys

    from tensordict.nn import TensorDictModule

    mixer_module = MIXERS[config.mixer](
        len(intention_models),
        config.intention_modeling.embedding_dim,
    ).to(device)
    if checkpoint["mixer"] is not None:
        mixer_module.load_state_dict(checkpoint["mixer"])
    embedding_mixer = TensorDictModule(
        mixer_module,
        in_keys=embedding_keys,
        out_keys=[("agent", "embedding")],
    )

    obs_key = [("agent", "observation"), ("agent", "embedding")]
    obs_dim = own_obs_dim + config.intention_modeling.embedding_dim

    main_agent = PPO(
        config=config.algorithm,
        n_agents=config.trainer.scenario.num_good_agents,
        group="agent",
        obs_dim=obs_dim,
        obs_key=obs_key,
        act_spec=env.full_action_spec_unbatched["agent", "action"],
        device=device,
        embedding_mixer=embedding_mixer,
    )
    main_agent.policy.load_state_dict(checkpoint["policy"])
    main_agent.policy.eval()
    for p in main_agent.policy.parameters():
        p.requires_grad_(False)

    return env, intention_model, main_agent


def collect(config, args, device):
    checkpoint_path = os.path.join(MODEL_DIR, args.experiment_name, "checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_envs = args.num_envs or config.trainer.num_test_envs
    env, intention_model, main_agent = build_and_load(
        config, checkpoint, device, num_envs, random_encoder=args.random_encoder
    )

    opponents = get_opponent_policies(
        config=config,
        train_env=env,
        test_env=env,
        pretrained_opponents_train=args.pretrained_opponents_train,
        pretrained_opponents_test=args.pretrained_opponents_test,
        group="adversary",
        obs_key=[("adversary", "observation")],
        num_train_envs=num_envs,
        num_test_envs=num_envs,
        device=device,
    )
    label_attr, num_classes = setup_opponent_labels(opponents)

    policy = TensorDictSequential(
        intention_model.process,
        main_agent.policy,
        opponents.policy,
        intention_model.postprocess,
    )

    max_steps = config.trainer.scenario.max_steps
    z_mix_rounds, z_rew_rounds, y_rounds = [], [], []

    with torch.no_grad():
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for r in range(args.num_rounds):
                opponents.reset_policies(r)
                intention_model.reset(num_envs)
                main_agent.reset(num_envs)

                rollout = env.rollout(
                    max_steps=max_steps,
                    policy=policy,
                    auto_cast_to_device=False,
                    break_when_any_done=False,
                    auto_reset=True,
                )
                T = rollout.batch_size[1]
                z_mix = rollout.get(Z_MIX_KEY).reshape(num_envs, T, -1)
                z_rew = rollout.get(Z_REWARD_KEY).reshape(num_envs, T, -1)
                z_mix_rounds.append(z_mix.cpu())
                z_rew_rounds.append(z_rew.cpu())
                y_rounds.append(getattr(opponents, label_attr).clone().cpu())
                print(f"  collected round {r + 1}/{args.num_rounds}")

    z_mix = torch.cat(z_mix_rounds, dim=0)
    z_rew = torch.cat(z_rew_rounds, dim=0)
    y = torch.cat(y_rounds, dim=0)
    return {"z_mix": z_mix, "z_reward": z_rew}, y, num_classes


def stratified_episode_split(y, test_frac, seed):
    rng = np.random.default_rng(seed)
    y_np = y.numpy()
    test_mask = np.zeros(len(y_np), dtype=bool)
    for cls in np.unique(y_np):
        idx = np.where(y_np == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_mask[idx[:n_test]] = True
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


def linear_probe(z, y, num_classes, train_ep, test_ep, args, device):
    E, T, D = z.shape

    def flatten(ep_idx):
        zt = z[ep_idx].reshape(-1, D)
        yt = y[ep_idx].unsqueeze(1).expand(-1, T).reshape(-1)
        return zt.to(device), yt.to(device)

    x_train, y_train = flatten(train_ep)
    x_test, y_test = flatten(test_ep)

    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = x_train.shape[0]
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            b = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(x_train[b]), y_train[b])
            loss.backward()
            opt.step()

    with torch.no_grad():
        test_pred = model(x_test).argmax(-1)
        test_acc = (test_pred == y_test).float().mean().item()

        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(y_test.cpu(), test_pred.cpu()):
            confusion[t, p] += 1

        z_test = ((z[test_ep].to(device) - mean) / std)
        per_t = []
        for t in range(T):
            pred_t = model(z_test[:, t, :]).argmax(-1).cpu()
            per_t.append((pred_t == y[test_ep]).float().mean().item())

    return test_acc, confusion, np.array(per_t)


def main():
    parser = argparse.ArgumentParser(description="Opponent-id linear probe")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="checkpoint at saved_models/<name>/checkpoint.pth",
    )
    parser.add_argument("--pretrained_opponents_train", type=str, required=True)
    parser.add_argument("--pretrained_opponents_test", type=str, default=None)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="parallel envs per round (default: config trainer.num_test_envs)",
    )
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="optional CSV path")
    parser.add_argument(
        "--random_encoder",
        action="store_true",
        help="skip loading trained encoder weights; probe a frozen, randomly "
        "initialized encoder of the same architecture instead, as a trajectory-"
        "flattening control",
    )
    args = parser.parse_args()

    if args.pretrained_opponents_test is None:
        args.pretrained_opponents_test = args.pretrained_opponents_train

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    device = config.device

    if args.random_encoder:
        print("Collecting embeddings (RANDOM, FROZEN encoder -- trajectory-flattening control)...")
    else:
        print("Collecting embeddings...")
    sources, y, num_classes = collect(config, args, device)

    train_ep, test_ep = stratified_episode_split(y, args.test_frac, args.seed)
    chance = 1.0 / num_classes
    counts = torch.bincount(y, minlength=num_classes)
    majority = counts.max().item() / counts.sum().item()

    print(f"\nopponents: {num_classes} | episodes: {len(y)} "
          f"(train {len(train_ep)}, test {len(test_ep)})")
    print(f"chance = {chance:.3f} | majority-class = {majority:.3f}\n")

    rows = []
    for name, z in sources.items():
        acc, confusion, per_t = linear_probe(
            z, y, num_classes, train_ep, test_ep, args, device
        )
        print(f"{name:>9}: test accuracy = {acc:.3f}  (chance {chance:.3f})")
        print(f"           confusion (rows=true, cols=pred):\n{confusion.numpy()}")
        rows.append((name, acc, per_t))

    if args.out:
        T = len(rows[0][2])
        with open(args.out, "w") as f:
            f.write("source,test_acc,chance," + ",".join(f"acc_t{t}" for t in range(T)) + "\n")
            for name, acc, per_t in rows:
                f.write(f"{name},{acc:.4f},{chance:.4f}," + ",".join(f"{a:.4f}" for a in per_t) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

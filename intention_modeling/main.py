import argparse
from tensordict.nn import TensorDictSequential, TensorDictModule

from intention_modeling.utils import load_config, save_model, save_checkpoint, apply_overrides
from intention_modeling.trainer import Trainer
from intention_modeling.rl_algorithms.ppo import PPO
from intention_modeling.opponents_handler import get_opponent_policies
from intention_modeling.mixers import MIXERS
from intention_modeling.intention_models.encoder_decoder import (
    EncoderDecoderIM,
    MultiHeadIM,
    EncoderDecoderList,
)
from intention_modeling.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="Load and run config from YAML file")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment/run"
    )
    parser.add_argument(
        "--intention_models",
        nargs="+",
        default=[],
        help="List of intention models to use",
    )
    parser.add_argument(
        "--shared_encoder",
        action="store_true",
        help="Ablation: train a single encoder with one auxiliary head per intention "
        "model, instead of one encoder per intention model mixed by a MoE",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save encoders, mixer and policy after training for downstream probing",
    )
    parser.add_argument(
        "--pretrained_opponents_train",
        default="",
        type=str,
        help="Pretrained opponents train set name",
    )
    parser.add_argument(
        "--pretrained_opponents_test",
        default="",
        type=str,
        help="Pretrained opponents test set name",
    )
    parser.add_argument(
        "--override",
        type=str,
        default="",
        help="Comma-separated config overrides, e.g. a.b.c=value,x.y=2",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.override)

    config.experiment_name = args.experiment_name
    config.intention_models = args.intention_models
    config.shared_encoder = args.shared_encoder

    logger = Logger(config)

    trainer = Trainer(
        config=config.trainer,
        logger=logger,
        device=config.device,
    )
    mixer = MIXERS[config.mixer]

    obs_key = [("agent", "observation")]
    obs_dim = trainer.env.observation_spec["agent", "observation"].shape[-1]
    embedding_mixer = None
    intention_models = []

    if len(args.intention_models) > 0:
        print(f"Modeling intentions using: {args.intention_models}")
        im_kwargs = dict(
            config=config.intention_modeling,
            own_obs_dim=trainer.env.observation_spec["agent", "observation"].shape[-1],
            opp_obs_dim=trainer.env.observation_spec["adversary", "observation"].shape[
                -1
            ],
            act_dim=trainer.env.full_action_spec["agent", "action"].shape[-1],
            n_opps=config.trainer.scenario.num_adversaries,
            device=config.device,
        )

        if args.shared_encoder:
            print("Sharing a single encoder across all intention modeling losses")
            intention_models.append(MultiHeadIM(models=args.intention_models, **im_kwargs))
        else:
            for model in args.intention_models:
                intention_models.append(EncoderDecoderIM(model=model, **im_kwargs))

        embedding_keys = []
        for intention_model in intention_models:
            embedding_keys += intention_model.process.out_keys

        obs_key += [("agent", "embedding")]
        obs_dim += config.intention_modeling.embedding_dim if config.mixer != "concat" else len(args.intention_models) * config.intention_modeling.embedding_dim
        embedding_mixer = TensorDictModule(
            mixer(len(intention_models), config.intention_modeling.embedding_dim).to(
                config.device
            ),
            in_keys=embedding_keys,
            out_keys=[("agent", "embedding")],
        )
        embedding_mixer.module.expert_names = [k[-1] for k in embedding_keys]
    else:
        print("Not modeling intentions!")
    intention_model = EncoderDecoderList(intention_models)

    main_agent = PPO(
        config=config.algorithm,
        n_agents=config.trainer.scenario.num_good_agents,
        group="agent",
        obs_dim=obs_dim,
        obs_key=obs_key,
        act_spec=trainer.env.full_action_spec_unbatched["agent", "action"],
        device=config.device,
        embedding_mixer=embedding_mixer,
    )

    pretrained_opponents = get_opponent_policies(
        config=config,
        train_env=trainer.env,
        test_env=trainer.test_env,
        pretrained_opponents_train=args.pretrained_opponents_train,
        pretrained_opponents_test=args.pretrained_opponents_test,
        group="adversary",
        obs_key=[("adversary", "observation")],
        num_train_envs=config.trainer.n_steps_per_iter
        // config.trainer.scenario.max_steps,
        num_test_envs=config.trainer.num_test_envs,
        device=config.device,
    )
    trainer.train(
        main_agent, pretrained_opponents, intention_model, pretrained_opponents=True
    )

    if args.save_model:
        meta = {
            "intention_models": args.intention_models,
            "shared_encoder": args.shared_encoder,
            "mixer": config.mixer,
            "embedding_dim": config.intention_modeling.embedding_dim,
            "hidden_dim": config.intention_modeling.hidden_dim,
            "own_obs_dim": trainer.env.observation_spec[
                "agent", "observation"
            ].shape[-1],
            "opp_obs_dim": trainer.env.observation_spec[
                "adversary", "observation"
            ].shape[-1],
            "act_dim": trainer.env.full_action_spec["agent", "action"].shape[-1],
            "n_opps": config.trainer.scenario.num_adversaries,
            "scenario": config.trainer.scenario.name,
        }
        save_checkpoint(
            config.experiment_name,
            intention_model,
            embedding_mixer,
            main_agent,
            meta,
        )


if __name__ == "__main__":
    main()

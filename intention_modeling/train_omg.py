import argparse
from tensordict.nn import TensorDictSequential, TensorDictModule

from intention_modeling.utils import load_config, save_model, apply_overrides
from intention_modeling.trainer import Trainer
from intention_modeling.rl_algorithms.ppo import PPO
from intention_modeling.opponents_handler import get_opponent_policies
from intention_modeling.mixers import MIXERS
from intention_modeling.intention_models.encoder_decoder import (
    EncoderDecoderIM,
    EncoderDecoderList,
)
from intention_modeling.intention_models.subgoal_inference import OMGIM
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
        "--pretrained_opponents_train",
        type=str,
        default="",
        help="Pretrained opponents train set name",
    )
    parser.add_argument(
        "--pretrained_opponents_test",
        type=str,
        default="",
        help="Pretrained opponents test set name",
    )
    parser.add_argument(
        "--override",
        type=str,
        default="",
        help="Comma-separated config overrides, e.g. a.b.c=value,x.y=2",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Name of pretrained vae. If not provided, will pretrain.",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.override)

    config.experiment_name = args.experiment_name

    logger = Logger(config)

    trainer = Trainer(
        config=config.trainer,
        logger=logger,
        device=config.device,
    )

    pretraining = args.vae is None
    obs_key = (
        [("agent", "observation"), ("agent", "embedding")]
        if not pretraining
        else [("agent", "observation")]
    )
    obs_dim = (
        trainer.env.observation_spec["agent", "observation"].shape[-1]
        + config.omg.embedding_dim
        if not pretraining
        else trainer.env.observation_spec["agent", "observation"].shape[-1]
    )
    mixer = MIXERS[config.mixer]
    embedding_mixer = None

    if not pretraining:
        embedding_mixer = TensorDictModule(
            mixer(1, config.omg.embedding_dim).to(config.device),
            in_keys=[("agent", "omg")],
            out_keys=[("agent", "embedding")],
        )

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

    omg_module = OMGIM(
        critic=getattr(main_agent, "eval_critic", main_agent.critic),
        n_iters=config.trainer.n_iters,
        obs_dim=trainer.env.observation_spec["agent", "observation"].shape[-1],
        act_dim=trainer.env.full_action_spec["agent", "action"].shape[-1],
        n_modelled_agents=config.trainer.scenario.num_adversaries,
        modelled_obs_dim=trainer.env.observation_spec["adversary", "observation"].shape[
            -1
        ],
        config=config.omg,
        device=config.device,
        vae=args.vae,
    )
    intention_model = EncoderDecoderList([omg_module])

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

    if pretraining:
        save_model(omg_module.vae, config.experiment_name)


if __name__ == "__main__":
    main()

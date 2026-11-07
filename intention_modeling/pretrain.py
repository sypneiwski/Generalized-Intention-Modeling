import argparse
from tensordict.nn import TensorDictSequential, TensorDictModule

from intention_modeling.utils import load_config, save_model
from intention_modeling.trainer import Trainer
from intention_modeling.rl_algorithms.ppo import PPO
from intention_modeling.rl_algorithms.ppo_diverse import PPO_Diverse
from intention_modeling.opponents_handler import PretrainedPoliciesHandler
from intention_modeling.intention_models.encoder_decoder import (
    EncoderDecoderIM,
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
        "--num_to_train",
        type=int,
        required=True,
        help="Num to train",
    )
    parser.add_argument(
        "--diversity_alpha",
        type=float,
        default=0.0,
        help="Diversity alpha",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    config.experiment_name = args.experiment_name
    logger = Logger(config)

    opponent_population = []
    for i in range(args.num_to_train):
        trainer = Trainer(
            config=config.trainer,
            logger=logger,
            device=config.device,
        )

        obs_key = [("agent", "observation")]
        obs_dim = trainer.env.observation_spec["agent", "observation"].shape[-1]
        embedding_mixer = None
        intention_model = EncoderDecoderList([])

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

        print(f"Pretraining! current={i}")
        opponents = PPO_Diverse(
            config=config.algorithm,
            n_agents=config.trainer.scenario.num_adversaries,
            group="adversary",
            obs_dim=trainer.env.observation_spec["adversary", "observation"].shape[-1],
            obs_key=[("adversary", "observation")],
            act_shape=trainer.env.full_action_spec["adversary", "action"].shape[-1],
            opponent_population=opponent_population,
            diversity_alpha=args.diversity_alpha,
            device=config.device,
        )
        trainer.train(main_agent, opponents, intention_model)
        save_model(opponents.policy, config.experiment_name)
        opponent_population.append(opponents.policy)


if __name__ == "__main__":
    main()

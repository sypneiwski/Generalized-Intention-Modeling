from intention_modeling.scenarios.custom_predator_prey import CustomPredatorPrey
from intention_modeling.scenarios.foraging_wrapper import ForagingWrapper
from intention_modeling.scenarios.kuhn_poker import KuhnWrapper
from intention_modeling.scenarios.football_wrapper import FootballWrapper

CUSTOM_SCENARIOS = {
    "custom_predator_prey": CustomPredatorPrey,
}

__all__ = ["CUSTOM_SCENARIOS", "ForagingWrapper", "KuhnWrapper", "FootballWrapper"]

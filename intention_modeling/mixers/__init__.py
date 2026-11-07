from intention_modeling.mixers.oned_mixer import (
    Top1Gate,
    ConcatMixer,
)

from intention_modeling.mixers.nd_mixer import (
    WeightedScoresMixerLNND,
)

MIXERS = {
    "lnnd": WeightedScoresMixerLNND,
    "top1": Top1Gate,
    "concat": ConcatMixer,
}

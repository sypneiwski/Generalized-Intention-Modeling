## Generalized Intention Modeling

Code for **Generalized Intention Modeling in Multi-Agent Reinforcement Learning**, accepted at NeurIPS 2026.

[[arXiv]](https://arxiv.org/abs/2605.31318) · [[OpenReview]](https://openreview.net/forum?id=ZSayYS6nBW)

### Setup

To install the dependencies, please create a virtual conda environment:

```
conda env create -f environment.yml
conda activate mixer_env
```

Install the dependencies with:

```
pip install "pip<24.1" setuptools==65.5.0 wheel==0.38.4 psutil
pip install -r requirements.txt --no-build-isolation
pip install -e .
```

#### Level-Based Foraging (LBF)

To run experiments for the Level-Based Foraging environment, you need to clone the original repository. In `intention_modeling/scenarios`, run:

```
git clone https://github.com/semitable/lb-foraging.git
cd lb-foraging
git checkout 79f383b7db0a3530f14252d5b6cc0d85aa9fb385
pip install .
```

#### Google Research Football (GRF)

To run experiments for the Google Research Football environment, you need to clone the original repository. We slightly modify the code to expose AI bot actions to enable opponent modeling via `intention_modeling.patch`. In `intention_modeling/scenarios`, run:

```
git clone https://github.com/google-research/football.git
cd football
git checkout 3d9e754720a95621bba6475c4d3b0d56fe919014
git apply ../intention_modeling.patch
mv ../academy_run_to_score_with_keeper_and_defenders.py gfootball/scenarios/
pip install .
```

#### Logging

For logging, we use Weights & Biases. Please create an account and export your API key with:

```
export WANDB_API_KEY=<your api key>
```

### Running Experiments

To run an experiment, execute:

```
OMP_NUM_THREADS=4 python intention_modeling/main.py --config <config file> --experiment_name <experiment name> --intention_models <desired episode components to model>
```

For example:
```
OMP_NUM_THREADS=4 python intention_modeling/main.py --config configs/gfootball.yaml --experiment_name experiment --intention_models ACT OPP_OBS FUTURE_OBS RTG
```

List of example config files are in the `configs/` folder.
There are four standard components to model:
- Opponent actions: ACT
- Opponent observations: OPP_OBS
- Future environment states: FUTURE_OBS
- Sum of future rewards: RTG
MIX uses all four. To use MIX, you need to set `--intention_models ACT OPP_OBS FUTURE_OBS RTG`. Leaving this flag unset will train a backbone PPO policy with no intention modeling (NOM baseline).

### POPP and LBF Experiments

To run experiments on POPP or LBF, you first need to pretrain a set of opponents. This can be done using:
```
OMP_NUM_THREADS=4 python intention_modeling/pretrain.py --config <config file> --experiment_name <experiment name> --num_to_train <number of opponents to train>
```
The command accepts an additional argument controlling the auxiliary diversity objective multiplier (default = 0, no explicit diversity). For example, to use diversity with a value of 0.01, run:
```
OMP_NUM_THREADS=4 python intention_modeling/pretrain.py --config <config file> --experiment_name <experiment name> --num_to_train <number of opponents to train> --diversity_alpha 0.01
```
After the run finishes, the opponent policies will be saved to `./saved_models/` under the given experiment name. Example pretrained opponent policies are provided in `saved_models/`.

You can then run an experiment with:
```
OMP_NUM_THREADS=4 python intention_modeling/main.py --config <config file> --pretrained_opponents_train <experiment name of pretraining run> --pretrained_opponents_test <experiment name of pretraining run> --experiment_name <experiment name> --intention_models <desired episode components to model>
```
Setting both `--pretrained_opponents_train` and `--pretrained_opponents_test` flags to the same value will result in the 'seen' setting from the paper. Setting them to different values will result in the 'unseen' setting.

E.g., to train MIX on the POPP environment in the seen setting with no explicit opponent diversity, run:
```
OMP_NUM_THREADS=4 python intention_modeling/main.py --config configs/predator_prey.yaml --pretrained_opponents_train POPP_seen_no_div --pretrained_opponents_test POPP_seen_no_div --experiment_name experiment --intention_models ACT OPP_OBS FUTURE_OBS RTG
```

### Kuhn Poker

To control the opponent visibility setting, use the `test_on_unseen` config option. We note that Kuhn Poker results exhibit high seed variance: individual seeds of any method, including MIX and LIAM, can occasionally fail to converge or collapse. We recommend comparing methods over at least 10 seeds.

### Baselines

To run experiments with the NOM baseline, do not set the intention_models flag.

To run experiments with the LIAM baseline (*Agent Modelling under Partial Observability for Deep Reinforcement Learning*, Papoudakis et al., NeurIPS 2021), set the intention_models flag to LIAM: `--intention_models LIAM`.

To run experiments with the MeLIBA baseline (*Deep Interactive Bayesian Reinforcement Learning via Meta-Learning*, Zintgraf et al., AAMAS 2021), run:
```
OMP_NUM_THREADS=4 python intention_modeling/train_meliba.py --config <config file> --experiment_name <experiment name>
```
(with pretrained opponent flags if the environment is POPP or LBF).

To run experiments with the OMG baseline (*Opponent Modeling based on Subgoal Inference*, Yu et al., NeurIPS 2024), run a pretraining run first:
```
OMP_NUM_THREADS=4 python intention_modeling/train_omg.py --config <config file> --experiment_name <experiment name>
```
(with pretrained opponent flags if the environment is POPP or LBF). This will pretrain and save a VAE model in `./saved_models/` (example pretrained models are provided in `saved_models/`). Then, run a full training run with:
```
OMP_NUM_THREADS=4 python intention_modeling/train_omg.py --config <config file> --experiment_name <experiment name> --vae <experiment name of VAE pretraining run>
```

### Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{odrowazsypniewski2026generalized,
  title     = {Generalized Intention Modeling in Multi-Agent Reinforcement Learning},
  author    = {Odrowaz-Sypniewski, Mateusz and Bayrooti, Jasmine and Shankar, Ajay and Prorok, Amanda},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=ZSayYS6nBW}
}

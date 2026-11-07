import torch
from tensordict.nn import TensorDictModule

from intention_modeling.utils import load_all_models


def get_opponent_policies(
    config,
    train_env,
    test_env,
    pretrained_opponents_train,
    pretrained_opponents_test,
    group,
    obs_key,
    num_train_envs,
    num_test_envs,
    device,
):
    if config.trainer.scenario.name.startswith("gfootball_"):
        return GFootballDummyOpponentsHandler()
    
    if config.trainer.scenario.name == "Kuhn":
        return KuhnHandler(
            config.test_on_unseen,
            group,
            obs_key,
            num_train_envs,
            num_test_envs,
            device,
        )

    return PretrainedPoliciesHandler(
        pretrained_opponents_train,
        pretrained_opponents_test,
        group,
        obs_key,
        num_train_envs,
        num_test_envs,
        device,
    )


class GFootballDummyOpponentsHandler:
    def __init__(self):
        self.explore_policy = TensorDictModule(lambda: None, in_keys=[], out_keys=[])
        self.policy = self.explore_policy

    def reset_policies(self, _):
        pass

    def update(self, rollouts):
        return {}

    def eval(self):
        return {}



class PretrainedPoliciesHandler:
    def __init__(
        self,
        pretrained_opponents_train,
        pretrained_opponents_test,
        group,
        obs_key,
        num_train_envs,
        num_test_envs,
        device,
    ):
        self.device = device
        self.group = group
        self.num_train_envs = num_train_envs
        self.num_test_envs = num_test_envs

        self.train_set_policies = load_all_models(pretrained_opponents_train, device)
        self.test_set_policies = load_all_models(pretrained_opponents_test, device)
        self.train_idx = torch.randint(
            0, len(self.train_set_policies), (num_train_envs,), device=self.device
        )
        self.test_idx = torch.randint(
            0, len(self.test_set_policies), (self.num_test_envs,), device=self.device
        )

        self.explore_policy = TensorDictModule(
            self.run_train_policies, in_keys=obs_key, out_keys=[(group, "action")]
        )
        self.policy = TensorDictModule(
            self.run_test_policies, in_keys=obs_key, out_keys=[(group, "action")]
        )

    def reset_policies(self, _):
        self.train_idx = torch.randint(
            0, len(self.train_set_policies), (self.num_train_envs,), device=self.device
        )
        self.test_idx = torch.randint(
            0, len(self.test_set_policies), (self.num_test_envs,), device=self.device
        )

    def eval(self):
        return {}

    def run_train_policies(self, observations):
        actions = []
        for policy in self.train_set_policies:
            with torch.no_grad():
                actions.append(policy(observations)[1])

        actions_tensor = torch.stack(
            actions
        )
        actions = actions_tensor[
            self.train_idx, torch.arange(self.num_train_envs, device=self.device), :, :
        ]
        return actions

    def run_test_policies(self, observations):
        actions = []
        for policy in self.test_set_policies:
            with torch.no_grad():
                actions.append(policy(observations)[1])

        actions_tensor = torch.stack(
            actions
        )
        actions = actions_tensor[
            self.test_idx, torch.arange(self.num_test_envs, device=self.device), :, :
        ]
        return actions

class KuhnHandler:
    def __init__(
        self,
        test_on_unseen,
        group,
        obs_key,
        num_train_envs,
        num_test_envs,
        device,
    ):
        self.device = device
        self.group = group
        self.obs_key = obs_key
        
        self.num_train_envs = num_train_envs
        self.num_test_envs = num_test_envs
        self.test_on_unseen = test_on_unseen

        self.opponents = torch.tensor([
            # seen
            [0.0, 1.0],
            [1.0, 0.0],
            [0.67, 1.0],
            [1.0, 0.67],
            [0.0, 0.17],
            [0.17, 0.0],

            # unseen
            [0.75, 0.15],
            [0.45, 0.90],
            [0.85, 0.55],
            [0.12, 0.28],
            [0.25, 0.05],
            [0.10, 0.65],
        ], dtype=torch.float32, device=device)

        self.train_heuristic_id = torch.zeros(
            num_train_envs, dtype=torch.long, device=device
        )
        self.test_heuristic_id = torch.zeros(
            num_test_envs, dtype=torch.long, device=device
        )

        self.explore_policy = TensorDictModule(
            self.run_train_heuristic, in_keys=obs_key, out_keys=[(group, "action")]
        )
        self.policy = TensorDictModule(
            self.run_test_heuristic, in_keys=obs_key, out_keys=[(group, "action")]
        )

    def reset_policies(self, _=None):
        self.train_heuristic_id.random_(0, 6)
        if self.test_on_unseen:
            self.test_heuristic_id.random_(6, 12)
        else:
            self.test_heuristic_id.random_(0, 6)

    def eval(self):
        return {}

    def run_train_heuristic(self, observations):
        return self._logic(observations, self.train_heuristic_id)

    def run_test_heuristic(self, observations):
        return self._logic(observations, self.test_heuristic_id)

    def _logic(self, observations, heuristic_id):
        observations = observations.squeeze(-2)
        batch_size = observations.shape[0]
        
        props = self.opponents[heuristic_id]
        eta = props[:, 0]
        xi = props[:, 1]
        card_vec = observations[:, 2:5]
        cards = card_vec.argmax(dim=-1)

        p1_passed = observations[:, 5] == 1.0
        p1_betted = observations[:, 6] == 1.0

        prob_bet = torch.zeros(batch_size, device=self.device)

        mask_pass = p1_passed
        if mask_pass.any():
            mask_j = mask_pass & (cards == 0)
            prob_bet[mask_j] = eta[mask_j]
            
            mask_k = mask_pass & (cards == 2)
            prob_bet[mask_k] = 1.0

        mask_bet = p1_betted
        if mask_bet.any():
            mask_q = mask_bet & (cards == 1)
            prob_bet[mask_q] = xi[mask_q]
            
            mask_k = mask_bet & (cards == 2)
            prob_bet[mask_k] = 1.0

        rand_vals = torch.rand(batch_size, device=self.device)
        actions_idx = (rand_vals < prob_bet).long()

        actions = torch.zeros((batch_size, 1, 2), device=self.device)
        actions.scatter_(2, actions_idx.unsqueeze(-1).unsqueeze(-1), 1.0)

        return actions

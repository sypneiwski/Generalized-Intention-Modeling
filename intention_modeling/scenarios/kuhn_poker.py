import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
)

class KuhnPokerEnv:
    def __init__(self):
        self.num_actions = 2
        self.cards = [0, 1, 2]
        self.deck = []
        self.player_cards = [None, None]
        self.history = []
        self.turn = 0
        self.pot_contrib = [1.0, 1.0]
        self.prev_opp_card = None

    def _get_obs(self, player_idx):
        obs = np.zeros(14, dtype=np.float32)
        obs[player_idx] = 1.0
        card = self.player_cards[player_idx]
        if card is not None:
            obs[2 + card] = 1.0
        for i, action in enumerate(self.history):
            base_idx = 5 + (i * 2)
            if action == 0: obs[base_idx] = 1.0
            else: obs[base_idx + 1] = 1.0
        if len(self.history) == 0 and self.prev_opp_card is not None:
            obs[11 + self.prev_opp_card[player_idx]] = 1.0
        return obs

    def reset(self):
        np.random.shuffle(self.cards)
        self.deck = self.cards[:]
        self.player_cards = [self.deck[0], self.deck[1]]
        
        self.history = []
        self.pot_contrib = [1.0, 1.0]
        self.turn = 0
        
        return (self._get_obs(0), self._get_obs(1)), {}

    def get_payoff(self):
        p1_card = self.player_cards[0]
        p2_card = self.player_cards[1]
        winner_idx = 0 if p1_card > p2_card else 1
        
        if self.history == [0, 0]:           # Pass, Pass
            return (1.0, -1.0) if winner_idx == 0 else (-1.0, 1.0)
        elif self.history == [0, 1, 0]:      # Pass, Bet, Pass (Fold)
            return (-1.0, 1.0)
        elif self.history == [0, 1, 1]:      # Pass, Bet, Bet (Call)
            return (2.0, -2.0) if winner_idx == 0 else (-2.0, 2.0)
        elif self.history == [1, 0]:         # Bet, Pass (Fold)
            return (1.0, -1.0)
        elif self.history == [1, 1]:         # Bet, Bet (Call)
            return (2.0, -2.0) if winner_idx == 0 else (-2.0, 2.0)
        return 0.0, 0.0

    def step(self, actions):
        active_player = self.turn
        active_action = actions[active_player]

        self.history.append(active_action)
        
        if active_action == 1:
            self.pot_contrib[active_player] += 1.0

        is_terminal = False
        
        if len(self.history) == 1:
            self.turn = 1
        elif len(self.history) == 2:
            if self.history[0] == 0 and self.history[1] == 1:
                self.turn = 0
            else:
                is_terminal = True
        elif len(self.history) == 3:
            is_terminal = True
            
        rewards = (0.0, 0.0)
        
        if is_terminal:
            rewards = self.get_payoff()
            self.prev_opp_card = self.player_cards[1], self.player_cards[0]
            new_obs, _ = self.reset()
            obs = new_obs
        else:
            obs = (self._get_obs(0), self._get_obs(1))

        return obs, rewards, False, False, {}

class KuhnWrapper:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.envs = [KuhnPokerEnv() for _ in range(num_envs)]
        self.group_map = {"agent": 1, "adversary": 1}

        self.observation_spec = CompositeSpec({
            "agent": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(shape=(14,))
            }),
            "adversary": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(shape=(14,))
            }),
        })

        self.full_action_spec = CompositeSpec({
            "agent": CompositeSpec({
                "action": OneHotDiscreteTensorSpec(n=2, shape=(1, 2))
            }),
            "adversary": CompositeSpec({
                "action": OneHotDiscreteTensorSpec(n=2, shape=(1, 2))
            }),
        })
        self.full_action_spec_unbatched = self.full_action_spec

    def _reset_all(self):
        agent_obs = []
        adv_obs = []
        for env in self.envs:
            env.prev_opp_card = None
            obs, _ = env.reset()
            agent_obs.append(obs[0])
            adv_obs.append(obs[1])
        return np.stack(agent_obs), np.stack(adv_obs)

    def rollout(self, max_steps, policy, callback=None, **kwargs):
        num_envs = self.num_envs
        agent_obs, adv_obs = self._reset_all()
        rollout_steps = []

        for t in range(max_steps):
            if callback: callback(self, None)

            td = TensorDict(
                source={
                    "agent": TensorDict(
                        source={"observation": torch.from_numpy(agent_obs).float().unsqueeze(1)},
                        batch_size=[num_envs, 1],
                    ),
                    "adversary": TensorDict(
                        source={"observation": torch.from_numpy(adv_obs).float().unsqueeze(1)},
                        batch_size=[num_envs, 1],
                    ),
                    "next": TensorDict(
                        source={
                            "agent": TensorDict(batch_size=[num_envs, 1]),
                            "adversary": TensorDict(batch_size=[num_envs, 1]),
                        },
                        batch_size=[num_envs]
                    ),
                },
                batch_size=[num_envs],
            )

            policy(td)

            agent_action = td.get(("agent", "action")).argmax(-1).squeeze(1).cpu().numpy()
            adv_action = td.get(("adversary", "action")).argmax(-1).squeeze(1).cpu().numpy()

            next_agent_obs = np.zeros_like(agent_obs)
            next_adv_obs = np.zeros_like(adv_obs)
            rewards = np.zeros((num_envs, 2), dtype=np.float32)
            done = np.zeros((num_envs, 1), dtype=bool)

            for i, env in enumerate(self.envs):
                (o_agent, o_adv), (r_agent, r_adv), d, _, _ = env.step(
                    (int(agent_action[i]), int(adv_action[i]))
                )
                next_agent_obs[i] = o_agent
                next_adv_obs[i] = o_adv
                rewards[i, 0] = r_agent
                rewards[i, 1] = r_adv
                done[i] = d if t != max_steps - 1 else True

            done_t = torch.from_numpy(done)
            td.set("done", done_t, inplace=True)
            td.set("terminated", done_t, inplace=True)
            td.set(("next", "done"), done_t, inplace=True)
            td.set(("next", "terminated"), done_t, inplace=True)

            rew_t = torch.from_numpy(rewards)
            td.set(("next", "agent", "reward"), rew_t[:, 0].view(num_envs, 1, 1), inplace=True)
            td.set(("next", "adversary", "reward"), rew_t[:, 1].view(num_envs, 1, 1), inplace=True)

            td.set(
                ("next", "agent", "observation"),
                torch.from_numpy(next_agent_obs).float().unsqueeze(1),
                inplace=True,
            )
            td.set(
                ("next", "adversary", "observation"),
                torch.from_numpy(next_adv_obs).float().unsqueeze(1),
                inplace=True,
            )

            rollout_steps.append(td)
            agent_obs, adv_obs = next_agent_obs, next_adv_obs

        return torch.stack(rollout_steps, dim=1)
    
    def render(self, *args, **kwargs):
        return None

if __name__ == "__main__":
    env = KuhnWrapper(num_envs=1)
    
    def random_policy(td):
        batch = td.shape[0]
        a_act = torch.nn.functional.one_hot(torch.randint(0, 2, (batch,)), num_classes=2)
        adv_act = torch.nn.functional.one_hot(torch.randint(0, 2, (batch,)), num_classes=2)
        
        td.set(("agent", "action"), a_act.unsqueeze(1))
        td.set(("adversary", "action"), adv_act.unsqueeze(1))

    print("Running 30 steps of random rollout...")
    env.rollout(max_steps=30, policy=random_policy)
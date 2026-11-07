# Adapted from https://github.com/uoe-agents/LIAM

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class CustomPredatorPrey(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_good_agents = kwargs.pop("num_good_agents", 1)
        num_adversaries = kwargs.pop("num_adversaries", 3)
        num_landmarks = kwargs.pop("num_landmarks", 2)
        self.agents_share_rew = kwargs.pop("agents_share_rew", False)
        self.adversaries_share_rew = kwargs.pop("adversaries_share_rew", True)
        self.observe_same_team = kwargs.pop("observe_same_team", True)
        self.observe_pos = kwargs.pop("observe_pos", True)
        self.observe_vel = kwargs.pop("observe_vel", True)
        self.bound = kwargs.pop("bound", 1.3)
        self.respawn_at_catch = kwargs.pop("respawn_at_catch", False)
        self.reset_callbacks = []
        self.device = device
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.visualize_semidims = False

        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=self.bound,
            y_semidim=self.bound,
            substeps=10,
            collision_force=500,
        )
        num_agents = num_adversaries + num_good_agents
        self.adversary_radius = 0.07

        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            name = f"adversary_{i}" if adversary else f"agent_{i - num_adversaries}"
            agent = Agent(
                name=name,
                collide=True,
                shape=Sphere(radius=0.07),
                u_multiplier=3.0 if adversary else 4.0,
                max_speed=1.0 if adversary else 1.3,
                color=Color.RED if adversary else Color.GREEN,
                adversary=adversary,
            )
            world.add_agent(agent)
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                shape=Sphere(radius=0.2),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def register_reset_callback(self, callback):
        self.reset_callbacks.append(callback)

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -0.9,
                    0.9,
                ),
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -0.9,
                    0.9,
                ),
                batch_index=env_index,
            )

        if env_index is None:
            for callback in self.reset_callbacks:
                callback(self.world.batch_dim)

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        dist_min = 2 * agent1.shape.radius + 2 * agent2.shape.radius
        return dist < dist_min

    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            for a in self.world.agents:
                a.rew = (
                    self.adversary_reward(a) if a.adversary else self.agent_reward(a)
                )

        if agent.adversary:
            return agent.rew
        else:
            return agent.rew

    def agent_reward(self, agent: Agent):

        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        num_collisions = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        adversaries = self.adversaries()

        if agent.collide:
            for a in adversaries:
                num_collisions[self.is_collision(a, agent)] += 1

        rew[num_collisions == 1] = 1
        rew[num_collisions > 1] -= 1

        mask = (
            (agent.state.pos[:, 0] > 0.99)
            | (agent.state.pos[:, 0] < -0.99)
            | (agent.state.pos[:, 1] < -0.99)
            | (agent.state.pos[:, 1] > 0.99)
        )
        rew[mask] -= 10
        return rew

    def adversary_reward(self, agent: Agent):

        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        agents = self.good_agents()
        adversaries = self.adversaries()
        for adv in adversaries:
            rew -= (
                0.1
                * torch.min(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                a.state.pos - adv.state.pos,
                                dim=-1,
                            )
                            for a in agents
                        ],
                        dim=-1,
                    ),
                    dim=-1,
                )[0]
            )
        num_collisions = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    num_collisions[self.is_collision(ag, adv)] += 1

        rew[num_collisions == 1] -= 0.5
        rew[num_collisions > 1] += 1
        return rew

    def observation(self, agent: Agent):
        entity_pos = []
        for entity in self.world.landmarks:
            if agent.adversary:
                entity_pos.append(entity.state.pos - agent.state.pos)
            else:
                dist = entity.state.pos - agent.state.pos
                mask = (
                    (dist[:, 0] > 0.5)
                    | (dist[:, 0] < -0.5)
                    | (dist[:, 1] > 0.5)
                    | (dist[:, 1] < -0.5)
                )
                dist[mask] = torch.tensor([-5.0, -5.0], device=self.device)
                entity_pos.append(dist)

        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            if agent.adversary and not other.adversary:
                other_pos.append(other.state.pos - agent.state.pos)
                other_vel.append(other.state.vel)
            elif not agent.adversary and not other.adversary and self.observe_same_team:
                other_pos.append(other.state.pos - agent.state.pos)
                other_vel.append(other.state.vel)
            elif not agent.adversary and other.adversary:
                dist = other.state.pos - agent.state.pos
                mask = (
                    (dist[:, 0] > 0.5)
                    | (dist[:, 0] < -0.5)
                    | (dist[:, 1] > 0.5)
                    | (dist[:, 1] < -0.5)
                )
                dist[mask] = torch.tensor([-5.0, -5.0], device=self.device)
                other_pos.append(dist)
            elif agent.adversary and other.adversary and self.observe_same_team:
                other_pos.append(other.state.pos - agent.state.pos)

        obs = torch.cat(
            [
                *([agent.state.vel] if self.observe_vel else []),
                *([agent.state.pos] if self.observe_pos else []),
                *entity_pos,
                *other_pos,
                *other_vel,
            ],
            dim=-1,
        )
        return obs

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        for i in range(4):
            geom = Line(
                length=2
                * ((self.bound - self.adversary_radius) + self.adversary_radius * 2)
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 0
                        else -self.bound - self.adversary_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 1
                        else -self.bound - self.adversary_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)

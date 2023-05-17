import gym
from model import AlienBot
import torch
from copy import deepcopy
from torch.distributions import Categorical
import numpy as np
from datetime import datetime
import os
import functools
import operator
import torch.multiprocessing as mp_torch

mp_torch.set_sharing_strategy("file_system")


class Experience:
    def __init__(self, state, next_state, action_prob, action, reward):
        self.state = state
        self.next_state = next_state
        self.action_prob = action_prob
        self.action = action
        self.reward = reward
        self.advantage = 0


def worker(
    worker_id,
    old_policy,
    env_fn,
    EPOCH=1000,
    HORIZON=1024,
    GAMMA=0.99,
    LAMBDA=0.95,
    sync_event=None,
    update_event=None,
    queue=None,
):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    for _ in range(EPOCH):
        if update_event is not None:
            update_event.wait()

        env = env_fn()
        state, info = env.reset()
        state = torch.from_numpy(state).float()
        action_distribution, value = old_policy(state.unsqueeze(0))

        tau = []
        deltas = []
        values = []

        for _ in range(HORIZON):
            action = torch.multinomial(action_distribution, 1).item()

            next_state, reward, terminated, truncated = env.step(action)[:4]
            next_state = torch.from_numpy(next_state).float()
            next_action_distribution, next_value = old_policy(next_state.unsqueeze(0))

            tau_i = Experience(
                state, next_state, action_distribution[0][action], action, reward
            )
            values.append(value)
            deltas.append(reward + GAMMA * next_value.detach() - value)
            tau.append(tau_i)

            value = next_value.detach()
            state = next_state
            action_distribution = next_action_distribution
            # TODO: check whether to detach

            if terminated:
                state, info = env.reset()
                state = torch.from_numpy(state).float()
                action_distribution, value = old_policy(state.unsqueeze(0))
                tau[-1].next_state = None

        cur_GAE = 0
        cur_delta_sum = 0
        for tau_i, delta, value in zip(
            reversed(tau), reversed(deltas), reversed(values)
        ):
            # ---------------------------- calculate advantage --------------------------- #
            cur_GAE = delta + GAMMA * LAMBDA * cur_GAE
            tau_i.advantage = cur_GAE

            # -------------------------- calculate total reward -------------------------- #
            if tau_i.next_state is None:
                cur_delta_sum = tau_i.reward - value.detach()
            else:
                cur_delta_sum = delta + GAMMA * cur_delta_sum
            tau_i.reward = cur_delta_sum + value.detach()

        env.close()
        if not queue:
            return tau
        else:
            queue.put(tau)
            sync_event.set()


class PPO_trainer:
    ALPHA = 0.5
    EPOCH = 500
    HORIZON = 10
    BATCH_SIZE = 128
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_PARAM = 0.1 * ALPHA
    LR = 3 * 10e-4 * ALPHA

    def __init__(
        self,
        policy,
        env_fn,
        num_workers=None,
        model_dir="./models",
        checkpoint_name="model.ckpt",
        save_all=False,
    ):
        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = mp_torch.cpu_count()

        self.env_fn = env_fn

        self.queue = mp_torch.Queue(maxsize=self.num_workers)
        self.sync_events = [mp_torch.Event() for _ in range(self.num_workers)]
        self.update_event = mp_torch.Event()
        self.processes = []
        self.policy = policy
        self.old_policy = deepcopy(policy)

        for param in self.old_policy.parameters():
            param.requires_grad = False

        self.model_dir = model_dir
        self.checkpoint_name = checkpoint_name
        self.save_all = save_all
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def tau_batch(self, tau, batch_size):
        return [tau[i : i + batch_size] for i in range(0, len(tau), batch_size)]

    def worker_init(self):
        for worker_id, sync_event in zip(range(self.num_workers), self.sync_events):
            # mp.set_start_method("spawn")

            p = mp_torch.Process(
                target=worker,
                args=(
                    worker_id,
                    self.old_policy,
                    self.env_fn,
                    self.EPOCH,
                    self.HORIZON,
                    self.GAMMA,
                    self.LAMBDA,
                    sync_event,
                    self.update_event,
                    self.queue,
                ),
            )
            p.start()
            self.processes.append(p)

    def tensor_transform(self, batch):
        states = torch.stack([tau_i.state for tau_i in batch])
        rewards = torch.tensor([tau_i.reward for tau_i in batch], dtype=torch.float32)
        advantages = torch.tensor(
            [tau_i.advantage for tau_i in batch], dtype=torch.float32
        )
        action_probs = torch.stack([tau_i.action_prob for tau_i in batch])
        actions = torch.tensor(
            [tau_i.action for tau_i in batch], dtype=torch.int64
        ).unsqueeze(1)

        return states, rewards, advantages, action_probs, actions

    def loss_CLIP_VF_S(
        self,
        states,
        rewards,
        advantages,
        action_probs,
        actions,
    ):
        theta_distributions, values = self.policy(states)
        theta_probs = theta_distributions.gather(1, actions).squeeze(1)
        ratios = torch.div(theta_probs, action_probs)
        unclipped_losses = ratios * advantages
        clipped_losses = (
            torch.clamp(ratios, 1 - self.CLIP_PARAM, 1 + self.CLIP_PARAM) * advantages
        )

        policy_losses = -torch.min(unclipped_losses, clipped_losses)
        v_losses = torch.nn.MSELoss()(values.squeeze(1), rewards)
        entropy_bonus = Categorical(probs=theta_distributions).entropy().mean()

        return policy_losses.mean() + v_losses.mean() - 0.01 * entropy_bonus

    def train_multiworker(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LR)
        self.worker_init()
        try:
            for epoch in range(self.EPOCH):
                print(f"Running EPOCH {epoch}")
                # ---------------------------------- collect --------------------------------- #
                self.update_event.set()
                for e in self.sync_events:
                    e.wait()
                    e.clear()
                self.update_event.clear()
                tau = functools.reduce(
                    operator.add,
                    (self.queue.get() for _ in range(self.num_workers)),
                    [],
                )
                # ---------------------------------- update ---------------------------------- #

                print(f"updating with {len(tau)} trajectories")
                for batch in self.tau_batch(tau, batch_size=self.BATCH_SIZE):
                    batch_info = self.tensor_transform(batch)
                    total_loss = self.loss_CLIP_VF_S(*batch_info)
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                self.old_policy.load_state_dict(self.policy.state_dict())
                if self.save_all:
                    torch.save(
                        self.old_policy.state_dict(),
                        f"{self.model_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    )
                else:
                    torch.save(
                        self.old_policy.state_dict(),
                        f"{self.model_dir}/{self.checkpoint_name}",
                    )
        except Exception as e:
            print(f"Exception occurred during training: {e}")
        # Make sure to terminate processes on completion or error
        finally:
            for p in self.processes:
                p.terminate()
                p.join()
            torch.save(
                self.old_policy.state_dict(),
                f"{self.model_dir}/{self.checkpoint_name}",
            )

    def train(self):
        if self.num_workers > 1:
            self.train_multiworker()
        else:
            optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LR)
            for _ in range(self.EPOCH):
                # ---------------------------------- collect --------------------------------- #
                tau = worker(
                    1,
                    self.old_policy,
                    self.env_fn,
                    self.EPOCH,
                    self.HORIZON,
                    self.GAMMA,
                    self.LAMBDA,
                )
                # ---------------------------------- update ---------------------------------- #
                for batch in self.tau_batch(tau, batch_size=self.BATCH_SIZE):
                    batch_info = self.tensor_transform(batch)
                    total_loss = self.loss_CLIP_VF_S(*batch_info)
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                self.old_policy.load_state_dict(self.policy.state_dict())
                if self.save_all:
                    torch.save(
                        self.old_policy.state_dict(),
                        f"{self.model_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    )
                else:
                    torch.save(
                        self.old_policy.state_dict(),
                        f"{self.model_dir}/{self.checkpoint_name}",
                    )


def env_fn():
    return gym.make("ALE/Alien-v5", render_mode="human")


def main():
    policy = AlienBot()
    # policy.load_state_dict(torch.load("./models/model.ckpt"))
    trainer = PPO_trainer(policy, env_fn, num_workers=1)
    trainer.train()


if __name__ == "__main__":
    main()

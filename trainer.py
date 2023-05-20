import gymnasium as gym
from model import AlienBot
import torch
from copy import deepcopy
from torch.distributions import Categorical
import numpy as np
from datetime import datetime
import os
import functools
from statistics import mean
import torch.multiprocessing as mp_torch
import matplotlib.pyplot as plt
from tqdm import tqdm

mp_torch.set_sharing_strategy("file_system")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    EPOCH=1024,
    HORIZON=1024,
    GAMMA=0.99,
    LAMBDA=0.95,
    update_event=None,
    Barrier=None,
    queue=None,
):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    for _ in range(EPOCH):
        if update_event is not None:
            update_event.wait()

        env = env_fn()
        state, info = env.reset()
        state = torch.from_numpy(state).float() / 255.0  # normalize
        action_distribution, value = old_policy(state.unsqueeze(0))

        tau = []
        deltas = []
        values = []

        for _ in range(HORIZON):
            action = torch.multinomial(action_distribution, 1).item()

            next_state, reward, terminated, truncated = env.step(action)[:4]
            next_state = torch.from_numpy(next_state).float() / 255.0
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

            if terminated or truncated:
                state, info = env.reset()
                state = torch.from_numpy(state).float() / 255.0
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
        if not Barrier:
            return tau
        else:
            queue.put(tau)
            Barrier.wait()


class PPO_trainer:
    ALPHA = 1
    EPOCH = 500
    HORIZON = 100
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
        self.update_event = mp_torch.Event()
        self.Barrier = torch.multiprocessing.Barrier(self.num_workers + 1)

        self.processes = []
        self.policy = policy
        self.old_policy = deepcopy(policy).to(torch.device("cpu")).eval()

        self.model_dir = model_dir
        self.checkpoint_name = checkpoint_name
        self.save_all = save_all
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def tau_batch(self, tau, batch_size):
        return [tau[i : i + batch_size] for i in range(0, len(tau), batch_size)]

    def worker_init(self):
        for worker_id in range(self.num_workers):
            # mp.set_start_method("spawn")
            sampling_policy = self.old_policy.to(torch.device("cpu"))
            p = mp_torch.Process(
                target=worker,
                args=(
                    worker_id,
                    sampling_policy,
                    self.env_fn,
                    self.EPOCH,
                    self.HORIZON,
                    self.GAMMA,
                    self.LAMBDA,
                    self.update_event,
                    self.Barrier,
                    self.queue,
                ),
            )
            p.start()
            self.processes.append(p)

    def tensor_transform(self, batch):
        states = torch.stack([tau_i.state.to(device) for tau_i in batch]).to(device)
        action_probs = torch.stack([tau_i.action_prob for tau_i in batch]).to(device)
        actions = (
            torch.tensor([tau_i.action for tau_i in batch], dtype=torch.int64)
            .unsqueeze(1)
            .to(device)
        )

        rewards = torch.tensor(
            [tau_i.reward for tau_i in batch], dtype=torch.float32
        ).to(device)
        advantages = torch.tensor(
            [tau_i.advantage for tau_i in batch], dtype=torch.float32
        ).to(device)
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )  # standardization

        return states, action_probs, actions, rewards, advantages

    def loss_CLIP_VF_S(
        self,
        states,
        action_probs,
        actions,
        rewards,
        advantages,
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

    def finish_handling(self, reward_plot):
        for p in self.processes:
            p.terminate()
            p.join()
        torch.save(
            self.old_policy.state_dict(),
            f"{self.model_dir}/{self.checkpoint_name}",
        )
        plt.plot(range(len(reward_plot)), reward_plot)
        plt.show()
        if not os.path.exists("./graphs"):
            os.mkdir("./graphs")
        plt.savefig(f"./graphs/graph_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    def anneal_parameter(self):
        self.ALPHA -= 1 / self.EPOCH
        self.CLIP_PARAM = 0.1 * self.ALPHA
        self.LR = 3 * 10e-4 * self.ALPHA

    def single_worker_collect(self):
        tau = worker(
            1,
            self.old_policy.to(torch.device("cpu")),
            self.env_fn,
            self.EPOCH,
            self.HORIZON,
            self.GAMMA,
            self.LAMBDA,
        )
        tau_reward = tau[0].reward.squeeze(1).item()
        return tau, tau_reward

    def multi_worker_collect(self):
        self.update_event.set()
        self.Barrier.wait()
        self.update_event.clear()

        def parse_worker(cur_info, next_tau):
            tau_list, reward_tensor = cur_info
            return tau_list + next_tau, torch.cat(
                (reward_tensor, next_tau[0].reward), dim=0
            )

        tau, tau_reward = functools.reduce(
            parse_worker,
            (self.queue.get() for _ in range(self.num_workers)),
            ([], torch.tensor([])),
        )
        tau_reward = tau_reward.squeeze(1).mean().item()

        return tau, tau_reward

    def train(self):
        single_worker = self.num_workers == 1
        if not single_worker:
            self.worker_init()
        try:
            max_reward = 0
            optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LR, eps=1e-5)
            reward_plot = []
            for _ in tqdm(range(self.EPOCH), desc="Epoch"):
                # ---------------------------------- collect --------------------------------- #
                tau, tau_reward = (
                    self.single_worker_collect()
                    if single_worker
                    else self.multi_worker_collect()
                )
                tqdm.write(f"Current reward: {tau_reward:.4f}")
                reward_plot.append(tau_reward)
                if tau_reward > max_reward:
                    torch.save(
                        self.old_policy.state_dict(),
                        f"{self.model_dir}/model_{self.checkpoint_name}_{tau_reward}.ckpt",
                    )
                    max_reward = tau_reward
                # ---------------------------------- update ---------------------------------- #
                for batch in self.tau_batch(tau, batch_size=self.BATCH_SIZE):
                    batch_info = self.tensor_transform(batch)
                    total_loss = self.loss_CLIP_VF_S(*batch_info)
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                self.anneal_parameter()
                self.old_policy.load_state_dict(self.policy.state_dict())
                if self.save_all:
                    torch.save(
                        self.old_policy.state_dict(),
                        f"{self.model_dir}/model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S').ckpt}",
                    )
                else:
                    torch.save(
                        self.old_policy.state_dict(),
                        f"{self.model_dir}/{self.checkpoint_name}.ckpt",
                    )
        except KeyboardInterrupt:
            print("Performing graceful shutdown...")
        except Exception as e:
            print(f"Exception occurred during training: {e}")
        finally:
            self.finish_handling(reward_plot)

    def inference(self, num_episodes=5):
        episode_count = 0
        env = self.env_fn()
        state, _ = env.reset()
        state = torch.from_numpy(state).float() / 255.0
        action_distribution, value = self.old_policy(state.unsqueeze(0))

        while episode_count <= num_episodes:
            action = torch.multinomial(action_distribution, 1).item()

            next_state, _, terminated, truncated = env.step(action)[:4]
            next_state = torch.from_numpy(next_state).float() / 255.0

            if terminated or truncated:
                episode_count += 1
                state, _ = env.reset()
                state = torch.from_numpy(state).float() / 255.0

            else:
                state = next_state

            action_distribution, _ = self.old_policy(state.unsqueeze(0))

        env.close()


def env_fn():
    return gym.make("ALE/Alien-v5", render_mode="human")


def main():
    policy = AlienBot().to(device)
    # policy.load_state_dict(torch.load(f"./model 112.ckpt"))
    trainer = PPO_trainer(policy, env_fn, num_workers=1, checkpoint_name="model")
    # trainer.train()
    trainer.inference()


if __name__ == "__main__":
    main()

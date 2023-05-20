import torch.nn as nn
import torch.nn.init as init


class AlienBot(nn.Module):
    def __init__(self, num_actions=18):
        super().__init__()
        self.conv = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            # Conv3
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            # Conv4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            # Conv5
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),  # 128 * 14 * 10
            # Conv6
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )  # 128 7 5

        self.actor = nn.Sequential(
            nn.Linear(128 * 7 * 5, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, num_actions),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(128 * 7 * 5, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Orthogonal Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.orthogonal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)

        action_probs = self.actor(x)
        state_values = self.critic(x)

        return action_probs, state_values

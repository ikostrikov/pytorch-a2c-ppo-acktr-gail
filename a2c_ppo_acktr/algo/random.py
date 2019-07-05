import torch
import torch.optim as optim

class RANDOM_AGENT():

    def __init__(self, actor_critic, value_loss_coef, entropy_coef, acktr=True):
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=0.0, eps=1.0)

    def update(self, rollouts):
        return torch.tensor(0), torch.tensor(0), torch.tensor(0)

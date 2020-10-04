from torch.nn import Linear, MSELoss, Module, functional as F
import torch.optim as optimizers
import torch as T

# Model class


class DQClassifier(Module):
    def __init__(self, alpha, input_dims, n_actions):
        super(DQClassifier, self).__init__()

        self.input_to_hidden_layer = Linear(*input_dims, n_actions * 128)
        self.hidden_to_output_layer = Linear(n_actions * 128, n_actions)

        self.optimizer = optimizers.Adam(self.parameters(), lr=alpha)
        self.loss = MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        layer1_output = F.relu(self.input_to_hidden_layer(data))
        y_pred = self.hidden_to_output_layer(layer1_output)

        return y_pred

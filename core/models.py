import torch
import torch.nn as nn

class DeepUserIntentPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_clusters=5):
        super(DeepUserIntentPredictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.intent_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_clusters),
            nn.Softmax(dim=1)
        )
        self.value_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        intent = self.intent_predictor(encoded)
        value = self.value_predictor(encoded)
        return intent, value

import torch
import torch.nn as nn
from .transformer import grad_reverse


class DomainDiscriminator(nn.Module):
    """MLP that predicts source (0) vs target (1) domain."""

    def __init__(self, feature_dim, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.discriminator(x)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network with gradient reversal."""

    def __init__(self, feature_extractor, num_classes=2, hidden_dim=256):
        super(DANN, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = feature_extractor._linear

        feature_dim = feature_extractor.feature_dim
        self.domain_discriminator = DomainDiscriminator(feature_dim, hidden_dim)

    def forward(self, source, target, alpha=1.0):
        # Extract features
        source_features = self.feature_extractor(source)
        target_features = self.feature_extractor(target)

        # Classification (source only)
        source_pred = self.classifier(source_features)

        # Domain discrimination with gradient reversal
        source_features_reversed = grad_reverse(source_features, alpha)
        target_features_reversed = grad_reverse(target_features, alpha)

        source_domain = self.domain_discriminator(source_features_reversed)
        target_domain = self.domain_discriminator(target_features_reversed)

        return source_pred, source_domain, target_domain

    def predict(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x):
        return self.feature_extractor(x)


def compute_lambda(epoch, total_epochs, gamma=10):
    """Lambda scheduling from DANN paper: 2 / (1 + exp(-gamma * p)) - 1"""
    import math

    p = epoch / total_epochs
    return 2 / (1 + math.exp(-gamma * p)) - 1

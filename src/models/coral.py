import torch
import torch.nn as nn


def coral_loss(source, target):
    d = source.size(1)

    # Source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # Target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # Frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss


class DeepCORAL(nn.Module):
    def __init__(self, feature_extractor, num_classes=2):
        super(DeepCORAL, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = feature_extractor._linear

    def forward(self, source, target=None):
        loss = torch.tensor(0.0, device=source.device)
        source_features = self.feature_extractor(source)
        if self.training and target is not None:
            target_features = self.feature_extractor(target)
            loss = coral_loss(source_features, target_features)
        source_pred = self.classifier(source_features)
        return source_pred, loss

    def predict(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x):
        return self.feature_extractor(x)

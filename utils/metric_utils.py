import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = round(self._data.total[key] / self._data.counts[key], 6)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        return torch.sum(pred == target).item() / len(target)


def macro_f(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        score = f1_score(target.cpu(), pred.cpu(), average="macro")
        return score

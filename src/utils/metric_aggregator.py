from collections import defaultdict
from collections.abc import Mapping
import torch

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

class MetricAggregator:
    def __init__(self):
        self.sum = nested_defaultdict()
        self.count = nested_defaultdict()

    def update(self, stats_dict, n=1):
        self._update_recursive(self.sum, self.count, stats_dict, n)

    def _update_recursive(self, sum_dict, count_dict, stats, n):
        for k, v in stats.items():
            if k.startswith("_"):
                continue
            if isinstance(v, torch.Tensor):
                v=v.item()
            if isinstance(v, Mapping):
                self._update_recursive(sum_dict[k], count_dict[k], v, n)
            else:
                sum_dict[k] = sum_dict.get(k, 0.0) + float(v) * n
                count_dict[k] = count_dict.get(k, 0) + n

    def compute(self):
        return self._compute_recursive(self.sum, self.count)

    def _compute_recursive(self, sum_dict, count_dict):
        out = {}
        for k in sum_dict.keys():
            if k.startswith("_"):
                continue
            if isinstance(sum_dict[k], Mapping):
                out[k] = self._compute_recursive(sum_dict[k], count_dict[k])
            else:
                out[k] = sum_dict[k] / max(count_dict[k], 1)
        return out

    def reset(self):
        self.sum = nested_defaultdict()
        self.count = nested_defaultdict()

    def print(self):
        stats = self.compute()
        line = self._compact_str(stats)
        print(line)
    def _compact_str(self, d, prefix=""):
        """
        Converts nested dicts to compact one-line strings.
        Example:
           total=1.23 | decomp=2.34 | group: a=1.0 b=2.0
        """
        parts = []
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if isinstance(v, Mapping):
                sub = " ".join(f"{sk}={sv:.4f}" for sk, sv in v.items())
                parts.append(f"{k}: {sub}")
            else:
                parts.append(f"{k}={v:.4f}")
        return " | ".join(parts)
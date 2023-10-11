from typing import Dict

import numpy as np


class Statistics:
    def __init__(self) -> None:
        pass


def get_stats(ort_output: np.ndarray, pt_output: np.ndarray) -> Dict:
    output = {}
    absolute_diff = np.abs(ort_output - pt_output)
    max_absolute_diff = np.max(absolute_diff)
    # index_max = np.unravel_index(absolute_diff.argmax(), absolute_diff.shape)
    mean_absolute_diff = np.mean(absolute_diff)
    output["max_absolute_diff"] = max_absolute_diff
    output["mean_absolute_diff"] = mean_absolute_diff
    output["min_absolute_diff"] = np.min(absolute_diff)
    return output

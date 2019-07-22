from dataclasses import dataclass
import numpy as np


@dataclass
class LabelsManager:
    path_in: str
    filename: str
    data: np.ndarray
    length: int

    @staticmethod
    def load_labels(file_path, file_name):

        labels = np.loadtxt(file_path, delimiter=',', ndmin=1, dtype=int)

        return LabelsManager(file_path, file_name, labels, labels.shape[0])


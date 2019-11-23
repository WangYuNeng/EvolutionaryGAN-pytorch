from abc import ABC, abstractmethod

from models.base_model import BaseModel
from data.base_dataset import BaseDataset


class BaseEvaluator(ABC):

    def __init__(self, opt, model: BaseModel, dataset: BaseDataset):
        self.opt = opt
        self.model = model
        self.device = model.get_device()
        self.dataset = dataset

    @abstractmethod
    def get_current_scores(self):
        pass
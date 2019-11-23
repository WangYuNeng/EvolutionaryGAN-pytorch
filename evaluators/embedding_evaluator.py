from .base_evaluator import BaseEvaluator


class EmbeddingEvaluator(BaseEvaluator):

    def __init__(self, opt, model, dataset):
        super().__init__(opt, model, dataset)

    def get_current_scores(self):
        return {}

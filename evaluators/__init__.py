from .image_evaluator import ImageEvaluator
from .embedding_evaluator import EmbeddingEvaluator


def get_evaluator(opt, **kwargs):
    if opt.dataset_mode == 'embedding':
        cls = EmbeddingEvaluator
    else:
        cls = ImageEvaluator
    return cls(opt, **kwargs)

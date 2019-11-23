def get_evaluator(opt, **kwargs):
    if opt.dataset_mode == 'embedding':
        from .embedding_evaluator import EmbeddingEvaluator
        cls = EmbeddingEvaluator
    else:
        from .image_evaluator import ImageEvaluator
        cls = ImageEvaluator
    return cls(opt, **kwargs)

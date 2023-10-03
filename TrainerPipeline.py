from transformers import Trainer
import numpy as np
class TrainerPipeline:
    trainer: Trainer
    def __init__(self, trainer: Trainer, tokenizer):
        self.trainer = trainer
        self.tokenizer = tokenizer

    def preprocess_function(self, examples, sentence1_key, sentence2_key, max_seq_length):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = self.tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)
        return result
    def __call__(self, dataset, is_regression=True, label_list=None, **kwargs):
        prediction_results = self.trainer.predict(dataset, metric_key_prefix="predict")
        ret = []
        if is_regression:
            predictions = np.squeeze(prediction_results.predictions)
            # return [
            #     {"label": "LABEL_0", "score": x} for x in predictions
            # ]
            for item in predictions:
                ret.append(
                    [{"label": "LABEL_0", "score": item}, ]
                )
        else:
            assert label_list is not None, "label_list must be provided for classification task"
            for item in prediction_results.predictions:
                ret.append(
                    [{"label": label_list[i], "score": x} for i, x in enumerate(item)]
                )
        return ret

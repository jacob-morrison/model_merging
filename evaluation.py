"""Scripts for evaluation of models."""
import datasets as hfds
import tensorflow as tf


def load_metric_for_glue_task(task: str):
    if task == 'sst-2':
        task = 'sst2'
    if task == 'sts-b':
        task = 'stsb'
    return hfds.load_metric("glue", task)


def evaluate_model(model, dataset: tf.data.Dataset, metric: hfds.Metric, mergeable_models):
    for model_input, gold_references in dataset:
        input_model_predictions = mergeable_models[0](model_input).logits
        input_model_predictions = tf.argmax(input_model_predictions, axis=-1)
        model_predictions = model(model_input).logits
        model_predictions = tf.argmax(model_predictions, axis=-1)
        print('model inputs:')
        print(model_input)
        print('input model predictions:')
        print(input_model_predictions)
        print('model predictions:')
        print(model_predictions)
        print('gold references:')
        print(gold_references)
        print()
        metric.add_batch(predictions=model_predictions, references=gold_references)
    return metric.compute()


def average_score(score):
    return sum(score.values()) / len(score.values())

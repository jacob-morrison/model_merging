"""Script for actually merging models."""
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/acd13578qu/data/.cache/huggingface'

from absl import app
from absl import flags
from absl import logging
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

import data
import evaluation
import hdf5_util
import merging

def set_flags():
    FLAGS = flags.FLAGS

    # TODO: Add descriptions to flags

    # The target model will be first.
    flags.DEFINE_list("models", None, "")
    flags.DEFINE_string("glue_task", None, "")

    flags.DEFINE_list("fishers", None, "")

    flags.DEFINE_bool("from_pt", True, "")

    flags.DEFINE_string("split", "validation", "")
    flags.DEFINE_integer("n_examples", 4096, "")
    flags.DEFINE_integer("batch_size", 32, "")
    flags.DEFINE_integer("sequence_length", 128, "")

    flags.DEFINE_integer("n_coeffs", 51, "")
    flags.DEFINE_enum("coeff_mode", "grid", ["grid", "random"], "")

    flags.DEFINE_float("fisher_floor", 1e-6, "")
    flags.DEFINE_bool("favor_target_model", True, "")
    flags.DEFINE_bool("normalize_fishers", True, "")


def load_models(models_list):
    models = []
    # for i, model_str in enumerate(FLAGS.models):
    for i, model_str in enumerate(models_list):
        model_str = os.path.expanduser(model_str)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_str, from_pt=FLAGS.from_pt
        )
        models.append(model)
        if i == 0:
            tokenizer = AutoTokenizer.from_pretrained(model_str)
    return models, tokenizer


def load_fishers(fishers_list):
    if not fishers_list:
        return None
    fishers = []
    for fisher_str in fishers_list:
        fisher_str = os.path.expanduser(fisher_str)
        fisher = hdf5_util.load_variables_from_hdf5(fisher_str, trainable=False)
        fishers.append(fisher)
    return fishers


def get_coeffs_set():
    n_models = len(FLAGS.models)
    if FLAGS.coeff_mode == "grid":
        assert n_models == 2
        return merging.create_pairwise_grid_coeffs(FLAGS.n_coeffs)
    elif FLAGS.coeff_mode == "random":
        return merging.create_random_coeffs(n_models, FLAGS.n_coeffs)
    else:
        raise ValueError


def get_best_results(results):
    return max(results, key=lambda r: evaluation.average_score(r.score))


def main(_):
    set_flags()
    run_merge(FLAGS.models, FLAGS.fishers, FLAGS.glue_task, False)

def run_merge(models_list, fishers_list, task, set_flags = True):
    if set_flags:
        set_flags()

    if fishers_list:
        assert len(fishers_list) == len(models_list)

    models, tokenizer = load_models(models_list)

    fishers = load_fishers(fishers_list)

    ds = data.load_glue_dataset(
        task=task,
        split=FLAGS.split,
        tokenizer=tokenizer,
        max_length=FLAGS.sequence_length,
    )
    ds = ds.take(FLAGS.n_examples).batch(FLAGS.batch_size)

    metric = evaluation.load_metric_for_glue_task(task)

    coefficients_set = get_coeffs_set()

    results = merging.merging_coefficients_search(
        models,
        coefficients_set=coefficients_set,
        dataset=ds,
        metric=metric,
        fishers=fishers,
        fisher_floor=FLAGS.fisher_floor,
        favor_target_model=FLAGS.favor_target_model,
        normalize_fishers=FLAGS.normalize_fishers,
    )

    best = get_best_results(results)
    print(80 * "*")
    print(" Best Merge")
    print(80 * "*")
    merging.print_merge_result(best)
    return best


if __name__ == "__main__":
    app.run(main)

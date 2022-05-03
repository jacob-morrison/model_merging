"""Script for actually merging models."""
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/acd13578qu/data/.cache/huggingface'

from absl import app
from absl import flags
from absl import logging
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, ViTModel

import data
import evaluation
import hdf5_util
import merging

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

def load_models(models_list, from_pt):
    models = []
    # for i, model_str in enumerate(FLAGS.models):
    for i, model_str in enumerate(models_list):
        model_str = os.path.expanduser(model_str)
        if 'google/vit-base-patch16-224-in21k' in model_str:
            model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_str, from_pt=from_pt
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


def get_coeffs_set(models, coeff_mode, n_coeffs):
    n_models = len(models)
    if coeff_mode == "grid":
        assert n_models == 2
        return merging.create_pairwise_grid_coeffs(n_coeffs)
    elif coeff_mode == "random":
        return merging.create_random_coeffs(n_models, n_coeffs)
    else:
        raise ValueError


def get_best_results(results):
    return max(results, key=lambda r: evaluation.average_score(r.score))


def main(_):
    run_merge(
        FLAGS.models,
        FLAGS.fishers,
        FLAGS.glue_task,
        FLAGS.from_pt,
        FLAGS.split,
        FLAGS.n_examples,
        FLAGS.batch_size,
        FLAGS.sequence_length,
        FLAGS.n_coeffs,
        FLAGS.coeff_mode,
        FLAGS.fisher_floor,
        FLAGS.favor_target_model,
        FLAGS.normalize_fishers
    )

def run_merge(
        models_list,
        fishers_list,
        task,
        from_pt = True,
        split = 'validation',
        n_examples = 4096,
        batch_size = 32,
        sequence_length = 128,
        n_coeffs = 51,
        coeff_mode = 'grid',
        fisher_floor = 1e-6,
        favor_target_model = True,
        normalize_fishers = True
    ):
    if fishers_list:
        assert len(fishers_list) == len(models_list)

    models, tokenizer = load_models(models_list, from_pt)

    fishers = load_fishers(fishers_list)

    ds = data.load_glue_dataset(
        task=task,
        split=split,
        tokenizer=tokenizer,
        max_length=sequence_length,
    )
    ds = ds.take(n_examples).batch(batch_size)

    metric = evaluation.load_metric_for_glue_task(task)

    coefficients_set = get_coeffs_set(models, coeff_mode, n_coeffs)

    results = merging.merging_coefficients_search(
        models,
        coefficients_set=coefficients_set,
        dataset=ds,
        metric=metric,
        fishers=fishers,
        fisher_floor=fisher_floor,
        favor_target_model=favor_target_model,
        normalize_fishers=normalize_fishers,
    )

    best = get_best_results(results)
    print(80 * "*")
    print(" Best Merge")
    print(80 * "*")
    merging.print_merge_result(best)
    return best


if __name__ == "__main__":
    app.run(main)

"""The code for actually performing the merge."""
import collections
from typing import Optional, Sequence
import datasets as hfds
import tensorflow as tf
import tensorflow_probability as tfp
import hf_util
import evaluation

MergeResult = collections.namedtuple("MergeResult", ["coefficients", "score"])


def print_merge_result(result: MergeResult):
    print(f"Merging coefficients: {result.coefficients}")
    print("Scores:")
    for name, value in result.score.items():
        print(f"  {name}: {value}")


def create_pairwise_grid_coeffs(n_weightings: int):
    n_weightings -= 2
    denom = n_weightings + 1
    weightings = [((i + 1) / denom, 1 - (i + 1) / denom) for i in range(n_weightings)]
    weightings = [(0.0, 1.0)] + weightings + [(1.0, 0.0)]
    weightings.reverse()
    return weightings


def create_random_coeffs(n_models: int, n_weightings: int, seed: Optional[int] = None):
    if seed is not None:
        tf.random.set_seed(seed)
    dist = tfp.distributions.Dirichlet(tf.ones([n_models]))
    return dist.sample(n_weightings, seed=seed).numpy().tolist()


def _merge_with_coeffs(
    output_variables: Sequence[tf.Variable],
    variables_to_merge: Sequence[Sequence[tf.Variable]],
    coefficients: Sequence[float],
    fishers=None,
    fisher_floor: float = 1e-6,
    favor_target_model=True,
    normalization_constants=None,
):
    n_models = len(variables_to_merge)
    assert len(coefficients) == n_models

    if fishers is None:
        fishers = n_models * [1.0]
    else:
        assert len(fishers) == n_models

    if normalization_constants is not None:
        assert len(normalization_constants) == n_models
        coefficients = [w / n for w, n in zip(coefficients, normalization_constants)]

    for i, var in enumerate(output_variables):
        lhs, rhs = [], []
        for j, (mvars, coeff, fisher) in enumerate(
            zip(variables_to_merge, coefficients, fishers)
        ):
            diag = fisher if isinstance(fisher, float) else fisher[i]
            if not favor_target_model or j == 0:
                diag = tf.maximum(diag, fisher_floor)
            mvar = mvars[i]
            tmp = coeff * diag
            lhs.append(tmp)
            rhs.append(tmp * mvar)
        rhs = tf.reduce_sum(rhs, axis=0)
        lhs = tf.reduce_sum(lhs, axis=0)
        var.assign(rhs / lhs)


# TODO: This is on the right track!!
# I think I need to be more fine-grained about merging each layer,
# i.e. merge individual pieces (manually?) instead of the whole layer's trainable variables at once
# or maybe I'm missing a small piece of how the algo works -- maybe I'm doing variables_to_merge wrong
# ^^^ this last point might actually be the problem?
def _merge_with_coeffs_roberta_and_vit(
    roberta_model,
    vit_model,
    # mergeable_models,
    coefficients: Sequence[float],
    fishers=None,
    fisher_floor: float = 1e-6,
    favor_target_model=True,
    normalization_constants=None,
):
    n_models = 2 #len(2)
    assert len(coefficients) == n_models

    if fishers is None:
        fishers = n_models * [1.0]
    else:
        assert len(fishers) == n_models

    if normalization_constants is not None:
        assert len(normalization_constants) == n_models
        coefficients = [w / n for w, n in zip(coefficients, normalization_constants)]

    # roberta_model = hf_util.clone_model(mergeable_models[0])
    # vit_model = mergeable_models[1]

    # for i, var in enumerate(output_variables):
    for roberta_layer, vit_layer in zip(
            roberta_model.layers[0].encoder.layer,
            vit_model.layers[0].encoder.layer
        ):
        # variables_to_merge = [roberta_layer.trainable_variables, vit_layer.trainable_variables]
        merging_vars_roberta = []
        merging_vars_vit = []
        output_variables = [] # roberta_layer.trainable_variables
        for roberta_var, vit_var in zip(roberta_layer.trainable_variables, vit_layer.trainable_variables):
            if roberta_var.shape == vit_var.shape:
                merging_vars_roberta.append(roberta_var)
                merging_vars_vit.append(vit_var)
                output_variables.append(roberta_var)
        variables_to_merge = [merging_vars_roberta, merging_vars_vit]
        for i, var in enumerate(output_variables):
            lhs, rhs = [], []
            for j, (mvars, coeff, fisher) in enumerate(
                zip(variables_to_merge, coefficients, fishers)
            ):
                diag = fisher if isinstance(fisher, float) else fisher[i]
                if not favor_target_model or j == 0:
                    diag = tf.maximum(diag, fisher_floor)
                tmp = coeff * diag
                rhs.append(tmp * mvars[i])
                lhs.append(tmp)
            rhs = tf.reduce_sum(rhs, axis=0)
            lhs = tf.reduce_sum(lhs, axis=0)
            var.assign(rhs / lhs)

def _merge_with_coeffs_roberta_and_vit_just_attention(
    roberta_model,
    vit_model,
    # mergeable_models,
    coefficients: Sequence[float],
    fishers=None,
    fisher_floor: float = 1e-6,
    favor_target_model=True,
    normalization_constants=None,
):
    n_models = 2 #len(2)
    assert len(coefficients) == n_models

    if fishers is None:
        fishers = n_models * [1.0]
    else:
        assert len(fishers) == n_models

    if normalization_constants is not None:
        assert len(normalization_constants) == n_models
        coefficients = [w / n for w, n in zip(coefficients, normalization_constants)]

    # roberta_model = hf_util.clone_model(mergeable_models[0])
    # vit_model = mergeable_models[1]

    # for i, var in enumerate(output_variables):
    for roberta_layer, vit_layer in zip(
            roberta_model.layers[0].encoder.layer,
            vit_model.layers[0].encoder.layer
        ):
        # variables_to_merge = [roberta_layer.trainable_variables, vit_layer.trainable_variables]
        merging_vars_roberta = []
        merging_vars_vit = []
        output_variables = [] # roberta_layer.trainable_variables
        for roberta_var, vit_var in zip(
                roberta_layer.attention.self_attention.key.trainable_variables,
                vit_layer.attention.self_attention.key.trainable_variables
            ):
            if roberta_var.shape == vit_var.shape:
                merging_vars_roberta.append(roberta_var)
                merging_vars_vit.append(vit_var)
                output_variables.append(roberta_var)
        variables_to_merge = [merging_vars_roberta, merging_vars_vit]
        for i, var in enumerate(output_variables):
            lhs, rhs = [], []
            for j, (mvars, coeff, fisher) in enumerate(
                zip(variables_to_merge, coefficients, fishers)
            ):
                diag = fisher if isinstance(fisher, float) else fisher[i]
                if not favor_target_model or j == 0:
                    diag = tf.maximum(diag, fisher_floor)
                tmp = coeff * diag
                rhs.append(tmp * mvars[i])
                lhs.append(tmp)
            rhs = tf.reduce_sum(rhs, axis=0)
            lhs = tf.reduce_sum(lhs, axis=0)
            var.assign(rhs / lhs)

        merging_vars_roberta = []
        merging_vars_vit = []
        output_variables = [] # roberta_layer.trainable_variables
        for roberta_var, vit_var in zip(
                roberta_layer.attention.self_attention.value.trainable_variables,
                vit_layer.attention.self_attention.value.trainable_variables
            ):
            if roberta_var.shape == vit_var.shape:
                merging_vars_roberta.append(roberta_var)
                merging_vars_vit.append(vit_var)
                output_variables.append(roberta_var)
        variables_to_merge = [merging_vars_roberta, merging_vars_vit]
        for i, var in enumerate(output_variables):
            lhs, rhs = [], []
            for j, (mvars, coeff, fisher) in enumerate(
                zip(variables_to_merge, coefficients, fishers)
            ):
                diag = fisher if isinstance(fisher, float) else fisher[i]
                if not favor_target_model or j == 0:
                    diag = tf.maximum(diag, fisher_floor)
                tmp = coeff * diag
                rhs.append(tmp * mvars[i])
                lhs.append(tmp)
            rhs = tf.reduce_sum(rhs, axis=0)
            lhs = tf.reduce_sum(lhs, axis=0)
            var.assign(rhs / lhs)


def _l2_norm_of_fisher(fisher):
    norm_const = tf.reduce_sum([tf.reduce_sum(tf.square(d)) for d in fisher])
    return tf.sqrt(norm_const)


def generate_merged_for_coeffs_set(
    mergeable_models,
    coefficients_set: Sequence[Sequence[float]],
    fishers=None,
    fisher_floor: float = 1e-6,
    favor_target_model=True,
    normalize_fishers=True,
):
    # Create the model to yield, then handle the norm_constants
    if normalize_fishers and fishers is not None:
        norm_constants = [_l2_norm_of_fisher(f) for f in fishers]
    else:
        norm_constants = None

    # # The first model in the list of mergeable models is the "target" model and
    # # the rest are "donor" models.
    # output_model = hf_util.clone_model(mergeable_models[0])
    # output_variables = hf_util.get_mergeable_variables(output_model)

    # variables_to_merge = [hf_util.get_mergeable_variables(m) for m in mergeable_models]

    # # Make sure that all of the variable lists contain exactly the same number
    # # of variables.
    # assert len({len(output_variables)} | set(len(v) for v in variables_to_merge)) == 1

    for coefficients in coefficients_set:
        # The first model in the list of mergeable models is the "target" model and
        # the rest are "donor" models.
        output_model = hf_util.clone_model(mergeable_models[0])
        output_variables = hf_util.get_mergeable_variables(output_model)

        variables_to_merge = [hf_util.get_mergeable_variables(m) for m in mergeable_models]

        # TODO: filter variables to merge
        for i in range(len(variables_to_merge)):
            pass
            

        # Make sure that all of the variable lists contain exactly the same number
        # of variables.

        # for v in variables_to_merge:
            # print(len(v))
        # for m in mergeable_models:
            # print(m.summary())
            # print(m.layers)
        # print(len(output_variables))
        # print(len({len(output_variables)} | set(len(v) for v in variables_to_merge)))
        
        merging_roberta_and_vit = False
        assert merging_roberta_and_vit or (len({len(output_variables)} | set(len(v) for v in variables_to_merge)) == 1)

        if merging_roberta_and_vit:
            # _merge_with_coeffs_roberta_and_vit(
            _merge_with_coeffs_roberta_and_vit_just_attention(
                output_model,
                mergeable_models[1],
                coefficients=coefficients,
                fishers=fishers,
                fisher_floor=fisher_floor,
                favor_target_model=favor_target_model,
                normalization_constants=norm_constants,
            )

        else:
            _merge_with_coeffs(
                output_variables,
                variables_to_merge,
                coefficients=coefficients,
                fishers=fishers,
                fisher_floor=fisher_floor,
                favor_target_model=favor_target_model,
                normalization_constants=norm_constants,
            )
        yield coefficients, output_model
        # print(mergeable_models[0].bert.encoder.layer[0])
        # print(output_model.encoder.layer[0])
        # break


def merging_coefficients_search(
    mergeable_models,
    coefficients_set: Sequence[Sequence[float]],
    dataset: tf.data.Dataset,
    metric: hfds.Metric,
    fishers=None,
    fisher_floor: float = 1e-6,
    favor_target_model=True,
    normalize_fishers=True,
    print_results=True,
):
    merged_models = generate_merged_for_coeffs_set(
        mergeable_models,
        coefficients_set,
        fishers,
        fisher_floor=fisher_floor,
        favor_target_model=favor_target_model,
        normalize_fishers=normalize_fishers,
    )
    results = []
    for coeffs, merged_model in merged_models:
        score = evaluation.evaluate_model(merged_model, dataset, metric, mergeable_models) # delete mergeable_models
        result = MergeResult(coefficients=coeffs, score=score)
        results.append(result)
        if print_results:
            print_merge_result(result)
    return results

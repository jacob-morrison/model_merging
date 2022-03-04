"""Script for actually merging models."""
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/acd13578qu/data/.cache/huggingface'

# from absl import app
# from absl import flags
# from absl import logging
# from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

import merge_and_evaluate
import merging

    # print(80 * "*")
    # print(" Best Merge")
    # print(80 * "*")
    # merging.print_merge_result(best)

    # merge_and_evaluate.run_merge(models_list, fishers_list, task)

models = [
    'yoshitomo-matsubara/bert-base-uncased-cola',
    'textattack/bert-base-uncased-SST-2',
    'textattack/bert-base-uncased-MRPC',
    'textattack/bert-base-uncased-QQP',
    'textattack/bert-base-uncased-QNLI',
    'textattack/bert-base-uncased-RTE'
]

fishers = [
    '../fishers/cola-default-settings.hdf5',
    '../fishers/sst-2-default-settings.hdf5',
    '../fishers/mrpc-default-settings.hdf5',
    '../fishers/qqp-default-settings.hdf5',
    '../fishers/qnli-default-settings.hdf5',
    '../fishers/rte-default-settings.hdf5'
]

tasks = [
    'cola',
    'sst-2',
    'mrpc',
    'qqp',
    'qnli',
    'rte'
]

for target_idx in range(len(models)):
    target_model = models[target_idx]
    task = tasks[target_idx]
    for intermediate_idx in range(len(models)):
        intermediate_model = models[intermediate_idx]
        result = merge_and_evaluate.run_merge([target_model, intermediate_model], None, task)

        print(80 * '*')
        print(' Target model: ' + target_model)
        print(' Intermediate model: ' + intermediate_model)
        merging.print_merge_result(result)
        print(80 * '*')
        print()

print('merging meta script finished:')


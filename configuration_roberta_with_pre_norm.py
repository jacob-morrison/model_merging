""" RoBERTa configuration"""
from collections import OrderedDict
from typing import Mapping

from ...onnx import OnnxConfig
from ...utils import logging
from ..bert.configuration_bert import BertConfig


logger = logging.get_logger(__name__)

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
    "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
    "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/config.json",
    "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/config.json",
    "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/config.json",
}


class RobertaConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.


    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The [`RobertaConfig`] class directly inherits [`BertConfig`]. It reuses the same defaults. Please check the parent
    class for more information.

    Examples:

    ```python
    >>> from transformers import RobertaConfig, RobertaModel

    >>> # Initializing a RoBERTa configuration
    >>> configuration = RobertaConfig()

    >>> # Initializing a model from the configuration
    >>> model = RobertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class RobertaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )
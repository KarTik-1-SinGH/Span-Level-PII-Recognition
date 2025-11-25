from transformers import AutoModelForTokenClassification, AutoConfig
import torch.nn as nn
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str,
                 dropout: float = 0.2,
                 freeze_layers: int = 0):
    """
    Creates a Token Classification model with configurable dropout and optional encoder layer freezing.
    Args:
        model_name: HuggingFace model name
        dropout: extra dropout to improve precision & reduce overfitting
        freeze_layers: freeze first N transformer layers to reduce latency
    """

    # Load config with updated dropout (applies to classifier + encoder)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config
    )

    # Optional: freeze lower encoder layers for speed
    if freeze_layers > 0 and hasattr(model, "base_model"):
        encoder = getattr(model, model.base_model_prefix).encoder
        for layer_idx in range(min(freeze_layers, len(encoder.layer))):
            for param in encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    return model

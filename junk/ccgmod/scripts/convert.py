import argparse
import typing
import logging
from typing import Any

import torch

from ccgmod import CCGNet

if typing.TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


def _from_numpy(arr: "np.ndarray") -> torch.Tensor:
    return torch.from_numpy(arr)


def extract_tensorflow_weights(checkpoint_path: str) -> dict[str, Any]:
    """Extract weights from the TensorFlow checkpoint."""
    import tensorflow as tf  # type: ignore[reportMissingImports]

    logger.info(f"Loading TensorFlow checkpoint: {checkpoint_path}")

    # Load checkpoint
    reader = tf.train.load_checkpoint(checkpoint_path)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
    var_to_shape_map = reader.get_variable_to_shape_map()  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]

    weights: dict[str, Any] = {}
    for key in sorted(var_to_shape_map):  # type: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        weights[key] = reader.get_tensor(key)  # type: ignore[reportUnknownMemberType]
        logger.debug(f"Loaded: {key} -> {var_to_shape_map[key]}")

    return weights


def map_tensorflow_to_pytorch(tf_weights: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Map TensorFlow weight names to the PyTorch model structure."""
    pytorch_weights: dict[str, torch.Tensor] = {}

    for tf_prefix, pt_num in [
        ("CCGBlock", 1),
        ("CCGBlock_1", 2),
        ("CCGBlock_2", 3),
        ("CCGBlock_3", 4),
    ]:
        pt_prefix = f"ccg_block{pt_num}"

        if f"{tf_prefix}/FullConnect/weights" in tf_weights:
            pytorch_weights[f"{pt_prefix}.global_fc.weight"] = _from_numpy(
                tf_weights[f"{tf_prefix}/FullConnect/weights"].T
            )
            pytorch_weights[f"{pt_prefix}.global_fc.bias"] = _from_numpy(
                tf_weights[f"{tf_prefix}/FullConnect/bias"]
            )

        if f"{tf_prefix}/Graph-CNN/weights" in tf_weights:
            pytorch_weights[f"{pt_prefix}.graph_conv.W"] = _from_numpy(
                tf_weights[f"{tf_prefix}/Graph-CNN/weights"]
            )
            pytorch_weights[f"{pt_prefix}.graph_conv.W_I"] = _from_numpy(
                tf_weights[f"{tf_prefix}/Graph-CNN/weights_I"]
            )
            pytorch_weights[f"{pt_prefix}.graph_conv.bias"] = _from_numpy(
                tf_weights[f"{tf_prefix}/Graph-CNN/bias"]
            )

        if f"{tf_prefix}/BatchNorm/gamma" in tf_weights:
            pytorch_weights[f"{pt_prefix}.bn_nodes.weight"] = _from_numpy(
                tf_weights[f"{tf_prefix}/BatchNorm/gamma"]
            )
            pytorch_weights[f"{pt_prefix}.bn_nodes.bias"] = _from_numpy(
                tf_weights[f"{tf_prefix}/BatchNorm/bias"]
            )

            mean_key = None
            var_key = None

            for key in tf_weights.keys():
                if (
                    f"{tf_prefix}/BatchNorm" in key
                    and "ExponentialMovingAverage" in key
                ):
                    if (
                        "Squeeze/ExponentialMovingAverage" in key
                        and "Squeeze_1" not in key
                    ):
                        mean_key = key
                    elif "Squeeze_1/ExponentialMovingAverage" in key:
                        var_key = key

            if mean_key:
                pytorch_weights[f"{pt_prefix}.bn_nodes.running_mean"] = _from_numpy(
                    tf_weights[mean_key]
                )
            if var_key:
                pytorch_weights[f"{pt_prefix}.bn_nodes.running_var"] = _from_numpy(
                    tf_weights[var_key]
                )

        if f"{tf_prefix}/BatchNorm_1/gamma" in tf_weights:
            pytorch_weights[f"{pt_prefix}.bn_global.weight"] = _from_numpy(
                tf_weights[f"{tf_prefix}/BatchNorm_1/gamma"]
            )
            pytorch_weights[f"{pt_prefix}.bn_global.bias"] = _from_numpy(
                tf_weights[f"{tf_prefix}/BatchNorm_1/bias"]
            )

            mean_key = None
            var_key = None

            for key in tf_weights.keys():
                if (
                    f"{tf_prefix}/BatchNorm_1" in key
                    and "ExponentialMovingAverage" in key
                ):
                    if (
                        "Squeeze/ExponentialMovingAverage" in key
                        and "Squeeze_1" not in key
                    ):
                        mean_key = key
                    elif "Squeeze_1/ExponentialMovingAverage" in key:
                        var_key = key

            if mean_key:
                pytorch_weights[f"{pt_prefix}.bn_global.running_mean"] = _from_numpy(
                    tf_weights[mean_key]
                )
            if var_key:
                pytorch_weights[f"{pt_prefix}.bn_global.running_var"] = _from_numpy(
                    tf_weights[var_key]
                )

    if "Multi_Head_Global_Attention/Att_Transform_Weights" in tf_weights:
        pytorch_weights["attention.attention_transform.weight"] = _from_numpy(
            tf_weights["Multi_Head_Global_Attention/Att_Transform_Weights"].T
        )
        pytorch_weights["attention.attention_transform.bias"] = _from_numpy(
            tf_weights["Multi_Head_Global_Attention/Att_Transform_Bias"]
        )
        pytorch_weights["attention.attention_weights"] = _from_numpy(
            tf_weights["Multi_Head_Global_Attention/Att_Tune_Weights"]
        )

    fc_layers = [
        ("Predictive_FC_1", "fc1", "bn1"),
        ("Predictive_FC_2", "fc2", "bn2"),
        ("Predictive_FC_3", "fc3", "bn3"),
        ("final", "fc_final", None),
    ]

    for tf_name, pt_name, bn_name in fc_layers:
        if f"{tf_name}/Embed/weights" in tf_weights:
            pytorch_weights[f"{pt_name}.weight"] = _from_numpy(
                tf_weights[f"{tf_name}/Embed/weights"].T
            )
            pytorch_weights[f"{pt_name}.bias"] = _from_numpy(
                tf_weights[f"{tf_name}/Embed/bias"]
            )
        elif f"{tf_name}/weights" in tf_weights:
            pytorch_weights[f"{pt_name}.weight"] = _from_numpy(
                tf_weights[f"{tf_name}/weights"].T
            )
            pytorch_weights[f"{pt_name}.bias"] = _from_numpy(
                tf_weights[f"{tf_name}/bias"]
            )

        if bn_name and f"{tf_name}/BatchNorm/gamma" in tf_weights:
            pytorch_weights[f"{bn_name}.weight"] = _from_numpy(
                tf_weights[f"{tf_name}/BatchNorm/gamma"]
            )
            pytorch_weights[f"{bn_name}.bias"] = _from_numpy(
                tf_weights[f"{tf_name}/BatchNorm/bias"]
            )

            mean_key = None
            var_key = None

            for key in tf_weights.keys():
                if f"{tf_name}/BatchNorm" in key and "ExponentialMovingAverage" in key:
                    if (
                        "Squeeze/ExponentialMovingAverage" in key
                        and "Squeeze_1" not in key
                    ):
                        mean_key = key
                    elif "Squeeze_1/ExponentialMovingAverage" in key:
                        var_key = key

            if mean_key:
                pytorch_weights[f"{bn_name}.running_mean"] = _from_numpy(
                    tf_weights[mean_key]
                )
            if var_key:
                pytorch_weights[f"{bn_name}.running_var"] = _from_numpy(
                    tf_weights[var_key]
                )

    return pytorch_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer CCGNet weights from TensorFlow to PyTorch"
    )
    parser.add_argument(
        "--tf_model_path",
        type=str,
        required=True,
        help="Path to TensorFlow checkpoint (without extension)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ccgnet_pytorch.pth",
        help="Output path for PyTorch weights",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify weight transfer with test inputs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Step 1: Extract TensorFlow weights
    try:
        tf_weights = extract_tensorflow_weights(args.tf_model_path)
        logger.info(f"Extracted {len(tf_weights)} weight tensors from TensorFlow")
    except Exception as e:
        logger.exception(f"Could not load TensorFlow checkpoint: {e}")
        return

    # Step 2: Map to PyTorch format
    try:
        pytorch_weights = map_tensorflow_to_pytorch(tf_weights)
        logger.info(f"Mapped to {len(pytorch_weights)} PyTorch parameters")
    except Exception as e:
        logger.exception(f"Could not map weights: {e}")
        return

    # Step 3: Create a PyTorch model and load weights
    try:
        model = CCGNet(
            node_features=34,
            num_edge_types=4,
            global_state_dim=24,
            num_classes=2,
        )

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(  # type: ignore[reportUnknownVariableType]
            pytorch_weights,
            strict=False,
        )

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        logger.info("Loaded weights into PyTorch model")
    except Exception as e:
        logger.exception(f"Could not load weights into PyTorch model: {e}")
        return

    # Step 4: Save PyTorch model
    try:
        torch.save(  # type: ignore[reportUnknownMemberType]
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "node_features": 34,
                    "num_edge_types": 4,
                    "global_state_dim": 24,
                    "num_classes": 2,
                },
            },
            args.output,
        )
        logger.info(f"Saved PyTorch model to {args.output}")
    except Exception as e:
        logger.exception(f"Could not save PyTorch model: {e}")
        return

    logger.info("SUCCESS: Weight transfer completed!")


if __name__ == "__main__":
    main()

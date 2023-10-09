import copy
import logging
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime
import torch
import torch.nn as nn


class PyTorchOnnxChecker:
    def __init__(
        self, absolute_tolerance: float = 1e-6, preserve_named_module_onnx: bool = False
    ) -> None:
        """_summary_

        Args:
            absolute_tolerance (float, optional): The absolute tolerance
            threshold for output differences in each layer.
            Defaults to 1e-6.
            preserve_named_module_onnx (bool, optional): Whether to preserve
            intermediately generated onnx files. Defaults to False.
        """
        self.absolute_tolerance = absolute_tolerance
        self._temp_onnx_path = "pytorch_onnx_checker_metadata"
        self._preserve_named_module_onnx = preserve_named_module_onnx
        self.logger = logging.getLogger(__name__)
        self._input_dict = {}
        # We need output of previous layer to compare with current layer
        self._module_to_onnxruntime_output = {}

        # statistics
        self._module_to_stats = {}

    @staticmethod
    def _deep_copy_model(pytorch_model: nn.Module) -> nn.Module:
        """_summary_

        Args:
            pytorch_model (nn.Module): _description_

        Returns:
            nn.Module: _description_
        """
        pytorch_model = copy.deepcopy(pytorch_model)
        pytorch_model.eval()
        return pytorch_model

    def _group_by_module_types(self) -> Dict[str, Dict]:
        """_summary_"""
        pass

    def check_named_modules(self, pytorch_model: nn.Module, input_tensor: torch.Tensor):
        """_summary_
        Args:
            pytorch_model (nn.Module): The PyTorch module to check.
        """
        if input_tensor.shape[0] != 1:
            raise ValueError("Right now, the checker only supports batch size of 1")

        # Create folder if not exists
        metdata_folder_path = Path(self._temp_onnx_path)
        metdata_folder_path.mkdir(exist_ok=True, parents=True)

        # Create deep copy because we need to add hooks
        pytorch_model = self._deep_copy_model(pytorch_model)

        # We want all input shapes
        for module_name, module in pytorch_model.named_modules():
            module.register_forward_hook(
                get_input_shape_forward_hook(self._input_dict, module_name)
            )
        # After adding hook, forward tensor to trigger hook
        pytorch_model(input_tensor)
        # Create another deep copy because need to remove hooks
        # Ideally, just removing hooks is more efficient, but because
        # I want to get this working quickly
        pytorch_model = self._deep_copy_model(pytorch_model)

        # Create models
        for module_name, module in pytorch_model.named_modules():
            # The upper module is empty
            if module_name in self._input_dict:
                input_data = self._input_dict[module_name]
                torch.onnx.export(
                    module,
                    input_data,
                    metdata_folder_path / f"{module_name}.onnx",
                    verbose=False,
                    input_names=[f"input_{i}" for i in range(len(input_data))],
                )

        # Compare
        for module_name, module in pytorch_model.named_modules():
            if module_name in self._input_dict:
                module.register_forward_hook(
                    onnx_validation_hook(metdata_folder_path / f"{module_name}.onnx", module_name)
                )

        # Call forward to trigger hook
        pytorch_model(input_tensor)

        # Clean up
        if not self._preserve_named_module_onnx:
            shutil.rmtree(metdata_folder_path, ignore_errors=False, onerror=None)


def get_stats(ort_output: np.ndarray, pt_output: np.ndarray) -> Dict:
    output = {}
    absolute_diff = np.abs(ort_output - pt_output)
    max_absolute_diff = np.max(absolute_diff)
    # index_max = np.unravel_index(absolute_diff.argmax(), absolute_diff.shape)
    mean_absolute_diff = np.mean(absolute_diff)
    output["max_absolute_diff"] = max_absolute_diff
    output["mean_absolute_diff"] = mean_absolute_diff
    output["min_absolute_diff"] = np.min(absolute_diff)
    return output


def process_dict(pt_output: Dict, ort_output: list):
    if list(pt_output.keys()):
        for i, (key, value) in enumerate(pt_output.items()):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
                pt_output[key] = value
            elif isinstance(value, Dict):
                pt_output[key] = process_dict(value, ort_output[i])
            elif not isinstance(value, np.ndarray):
                raise ValueError(f"Unsupported type: {type(value)}")
            # ORT iterates by index
            get_stats(ort_output[i], value)


@torch.no_grad()
def onnx_validation_hook(onnx_runtime_path: str, module_name: str):
    def inner(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
        """The onnx validation hook.

        Args:
            module (nn.Module): _description_
            inputs (Tuple[torch.Tensor]): _description_
            outputs (torch.Tensor): For simplicity, let's assume
            that all outputs are tensors
        """
        # load model
        ort_model = onnxruntime.InferenceSession(onnx_runtime_path)

        # compute ONNX Runtime output prediction
        # Compare with pytorch module
        ort_output = ort_model.run(
            None,
            {
                input_data.name: input_tensor.detach().cpu().numpy()
                for input_data, input_tensor in zip(ort_model.get_inputs(), inputs)
            },
        )
        if isinstance(outputs, Dict):
            process_dict(outputs, ort_output)

        elif isinstance(outputs, torch.Tensor):
            pt_output = outputs.detach().cpu().numpy()
            stats = get_stats(ort_output, pt_output)
        print(f"Module name: {module_name}, stats: {stats}")

        # output = {}
        # output_dict[name] = output
        # output["mean"] = param.mean().detach().item()
        # output["std"] = param.std().detach().item()
        # output["max"] = param.max().detach().item()
        # output["min"] = param.min().detach().item()

    return inner


@torch.no_grad()
def get_input_shape_forward_hook(input_dict: Dict[str, torch.Tensor], module_name: str):
    def inner(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
        """_summary_

        Args:
            module (nn.Module): _description_
            inputs (Tuple[torch.Tensor]): _description_
            outputs (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # Get access to "self inside onnv validation hook"
        new_inputs = []
        for input_item in inputs:
            if isinstance(input_item, torch.Tensor):
                input_item = input_item.detach().cpu()
                new_inputs.append(torch.randn(*input_item.shape))

        # To prevent global module lock
        input_data = (
            new_inputs[0]
            if len(new_inputs) == 1
            else (input_data for input_data in new_inputs.values())
        )
        input_dict[module_name] = input_data

    return inner

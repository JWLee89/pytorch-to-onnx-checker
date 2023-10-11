import copy
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

from pt2onnx_checker.stats import get_stats


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

    def get_check_results(self) -> Dict[str, Any]:
        """Get the check results.

        Returns:
            Dict[str, Any]: The check results.
        """
        return self._module_to_stats

    def check_named_modules(self, pytorch_model: nn.Module, input_tensor: torch.Tensor) -> None:
        """Run a scan across all inner nn.Modules and create onnx representations for each module.

        Then, compare
        the outputs from onnxruntime forward vs pytorch and compute statistics.
        These statistics can provide a good indication on which operation shows
        the largest difference.
        Args:
            pytorch_model (nn.Module): The PyTorch module to check.
        """
        if input_tensor.shape[0] != 1:
            raise ValueError("Right now, the checker only supports batch size of 1")

        # Create folder if not exists
        metdata_folder_path = Path(self._temp_onnx_path)
        metdata_folder_path.mkdir(exist_ok=True, parents=True)

        # Create deep copy because we need to add hooks
        pytorch_model = copy.deepcopy(pytorch_model)
        pytorch_model.eval()

        # We want all input shapes
        for module_name, module in pytorch_model.named_modules():
            module.register_forward_hook(
                get_dummy_input_forward_hook(self._input_dict, module_name)
            )
        # After adding hook, forward tensor to trigger hook
        pytorch_model(input_tensor)
        # Create another deep copy because need to remove hooks
        # Ideally, just removing hooks is more efficient, but because
        # I want to get this working quickly
        pytorch_model = copy.deepcopy(pytorch_model)

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
                    onnx_validation_hook(
                        metdata_folder_path / f"{module_name}.onnx",
                        module_name,
                        self._module_to_stats,
                    )
                )

        # Call forward to trigger hook
        pytorch_model(input_tensor)

        # Clean up
        if not self._preserve_named_module_onnx:
            shutil.rmtree(metdata_folder_path, ignore_errors=False, onerror=None)


def convert_entries_to_np_array(
    pt_output: Dict, ort_output: List[np.ndarray], output_list: List[np.ndarray]
):
    if any(pt_output):
        for i, value in enumerate(pt_output.values()):
            # Convert tensors to numpy
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
                output_list.append(value)
            elif isinstance(value, Dict):
                convert_entries_to_np_array(value, ort_output[i])
            elif not isinstance(value, np.ndarray):
                raise ValueError(f"Unsupported type: {type(value)}")


def get_ort_inferencer(tensor: torch.Tensor, onnx_file_path: str) -> onnxruntime.InferenceSession:
    """Get the onnx runtime inference session. The execution provider is based on the device of the
    tensor passed in. If a cuda tensor is passed in,we will use CUDAExecutionProvider.

    Args:
        tensor (torch.Tensor): The pytorch tensor used to determine
        execution provider
        onnx_file_path (str): The location of the onnx file

    Returns:
        onnxruntime.InferenceSession: onnxruntime object used for inference
    """
    device = "CUDA" if tensor.is_cuda else "CPU"
    providers = [f"{device}ExecutionProvider"]
    return onnxruntime.InferenceSession(onnx_file_path, providers=providers)


@torch.no_grad()
def onnx_validation_hook(
    onnx_file_path: str, module_name: str, module_name_to_stats_dict: Dict[str, Any]
) -> Callable:
    """The validation logic that runs during forward pass, comparing PyTorch and OnnxRuntime
    outputs.

    Args:
        onnx_file_path (str): The location of the onnx file to load
        module_name (str): The current PyTorch module name + call count with delimiter
        module_name_to_stats_dict (Dict): The dictionary that stores the statistics
    """

    def inner(_: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
        """The onnx validation hook.

        Args:
            inputs (Tuple[torch.Tensor]): A tuple of input tensors
            outputs (torch.Tensor): For simplicity, let's assume
            that all outputs are tensors
        """
        ort_inferencer = get_ort_inferencer(outputs, onnx_file_path)

        # compute ONNX Runtime output prediction
        # Compare with pytorch module
        ort_output = ort_inferencer.run(
            None,
            {
                input_data.name: input_tensor.detach().cpu().numpy()
                for input_data, input_tensor in zip(ort_inferencer.get_inputs(), inputs)
            },
        )

        # Some models can output dictionaries.
        if isinstance(outputs, Dict):
            np_converted_outputs = []
            convert_entries_to_np_array(outputs, ort_output, np_converted_outputs)
        elif isinstance(outputs, torch.Tensor):
            np_converted_outputs = outputs.detach().cpu().numpy()

        stats = get_stats(ort_output, np_converted_outputs)
        if module_name in module_name_to_stats_dict:
            module_name_to_stats_dict[module_name].append(stats)
        else:
            module_name_to_stats_dict[module_name] = [stats]

    return inner


@torch.no_grad()
def get_dummy_input_forward_hook(
    input_dict: Dict[str, torch.Tensor], module_name: str
) -> Callable:
    """Store torch.randn tensors that are later on used to infer.

    Args:
        input_dict (Dict[str, torch.Tensor]): An input dict containing the
        following values:
            key: module name
            value: torch.randn tensor
        module_name (str): The name of the current module being evaluated

    Returns:
        Callable: A function that will be called during forward pass.
    """

    def inner(_: nn.Module, inputs: Tuple[torch.Tensor], __: torch.Tensor) -> None:
        """During the forward pass, if the input is a tensor, we will create random tensors with
        the same shape as the input tensor. This is needed when calling the torch.onnx.export
        function.

        Args:
            inputs (Tuple[torch.Tensor]): An input dictionary with string keys
            and tensor values.

        Returns: None
        """
        # Get access to "self inside onnv validation hook"
        new_inputs = []
        for input_item in inputs:
            if isinstance(input_item, torch.Tensor):
                input_item = input_item.detach().cpu()
                new_inputs.append(torch.randn(*input_item.shape))

        # To prevent global module lock, we use random tensors,
        # not the actual input tensors used in the forward pass
        input_data = (
            new_inputs[0]
            if len(new_inputs) == 1
            else (input_data for input_data in new_inputs.values())
        )
        input_dict[module_name] = input_data

    return inner

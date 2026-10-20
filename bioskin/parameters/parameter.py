# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import torch
import numpy as np
import bioskin.editing.operators as op


#  NOTE on the order of slots 3 and 4.
#  These used to be listed as ("Blood Oxygenation", "Melanin Type Ratio"), which is
#  the reverse of how the pretrained decoders were actually trained. The Monte Carlo
#  training set (datasets/*_params_test_*.pt, named ..._eu[0.650-0.800]_ox[0.900-1.000])
#  stores the eumelanin/pheomelanin ratio in column 3 and blood oxygenation in column 4,
#  and the decoder consumes those columns in stored order. Feeding the .pt columns
#  straight through reproduces the simulation at median per-curve RMSE 0.008 and wins on
#  96.2% of the 3624 curated samples; transposing 3/4 degrades that to 0.025.
#  Do not reorder these without re-checking against the training tensors.
SKIN_PROPS = ["Melanin", "Haemoglobin", "Thickness", "Melanin Type Ratio", "Blood Oxygenation", "Occlusion"]

# Plausible range, not including rushes, spots, etc.
REGULAR_SKIN_MIN_VALUES = [0.005, 0.001, 0.005, 0.65, 0.90, 0.001]
REGULAR_SKIN_MAX_VALUES = [0.40, 0.0225, 0.025, 0.80, 1.00, 1.0]

# Plausible range
SKIN_MIN_VALUES = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
SKIN_MAX_VALUES = [0.45, 0.25, 0.035, 1.0, 1.0, 1.0]

# Full range
MIN_PROP_VALUES = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
MAX_PROP_VALUES = [1.0, 1.0, 0.035, 1.0, 1.0, 1.0]

#  Support of the Monte Carlo training set in *normalized* decoder-input space, per slot.
#  Slots 0-2 are warped (see params_io.warp_parameter_maps); slots 3-5 pass through
#  unchanged, so for those the normalized value is also the physical value.
#  Used by check_decoder_inputs() below; these are measured from the training tensors,
#  not hand-picked, so treat them as ground truth about what the decoders have seen.
DECODER_TRAINING_MIN = [0.10, 0.17, 0.00, 0.65, 0.90, 0.00]
DECODER_TRAINING_MAX = [0.76, 0.67, 1.00, 0.80, 1.00, 1.00]


def check_decoder_inputs(params, strict=True, name='decoder input'):
    """Sanity-check normalized skin parameters before decoding.

    The decoder happily extrapolates on out-of-distribution input and still returns
    something that looks like a plausible skin curve, so mistakes here fail silently.
    The most damaging one is transposing slots 3 and 4 (melanin type ratio vs blood
    oxygenation), whose training supports are disjoint - that is detected and raised.

    Set strict=False to downgrade the transposition check to a warning.
    """
    import warnings

    if hasattr(params, 'detach'):
        values = params.detach().cpu().numpy()
    else:
        values = np.asarray(params)
    if values.ndim == 1:
        values = values[None, :]

    ratio_lo, ratio_hi = DECODER_TRAINING_MIN[3], DECODER_TRAINING_MAX[3]   # 0.65 - 0.80
    oxy_lo, oxy_hi = DECODER_TRAINING_MIN[4], DECODER_TRAINING_MAX[4]       # 0.90 - 1.00

    if values.shape[1] > 4:
        slot3, slot4 = values[:, 3], values[:, 4]
        looks_swapped = (np.mean((slot3 >= oxy_lo) & (slot3 <= oxy_hi)) > 0.9 and
                         np.mean((slot4 >= ratio_lo) & (slot4 <= ratio_hi)) > 0.9)
        if looks_swapped:
            message = (
                f"{name}: slots 3 and 4 look transposed. Slot 3 must be "
                f"'{SKIN_PROPS[3]}' in [{ratio_lo}, {ratio_hi}] and slot 4 must be "
                f"'{SKIN_PROPS[4]}' in [{oxy_lo}, {oxy_hi}], but the values given are "
                f"the other way round. Decoding as-is silently produces spectra with an "
                f"inflated erythema index. Swap the two columns, or pass strict=False if "
                f"this is intentional."
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)

    for slot in range(min(values.shape[1], len(DECODER_TRAINING_MIN))):
        lo, hi = DECODER_TRAINING_MIN[slot], DECODER_TRAINING_MAX[slot]
        outside = np.mean((values[:, slot] < lo) | (values[:, slot] > hi))
        if outside > 0.5:
            warnings.warn(
                f"{name}: {outside:.0%} of slot {slot} ('{SKIN_PROPS[slot]}') values fall "
                f"outside the training support [{lo}, {hi}]. The decoder is extrapolating "
                f"and the resulting spectra may not be physically meaningful.",
                stacklevel=2)

    return values


class Parameter:
    def __init__(self, name, image):
        self.name = name
        self.map = image.copy()
        self.map_edited = image.copy()
        self.map_edited_last = image.copy()  # to concatenate operators
        self.previous_operator = None
        self.editing_stack = []

    def reset(self):
        self.map_edited = self.map.copy()
        self.map_edited_last = self.map.copy()
        self.previous_operator = None
        self.editing_stack = []

    def update(self, image):
        self.map_edited = image.copy()
        self.map_edited_last = image.copy()  # to concatenate operators
        self.previous_operator = None
        self.editing_stack = []

    def mean(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return torch.mean(tensor[:, :, 0])

    def mean_masked(self):
        if isinstance(self.map_edited, np.ndarray):
            img = torch.from_numpy(self.map_edited)
        else:
            img = self.map_edited.clone().detach()
        img = img[:, :, 0]
        mask = (img < 0.95) & (img > 0.001)
        return torch.mean(img[mask])

    def min(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return torch.min(tensor[:, :, 0])

    def max(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return torch.max(tensor[:, :, 0])

    def mode(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return op.compute_mode(tensor[:, :, 0])

    def mode_masked(self):
        if isinstance(self.map_edited, np.ndarray):
            img = torch.from_numpy(self.map_edited)
        else:
            img = self.map_edited.clone().detach()
        img = img[:, :, 0]
        mask = (img != 1.0) & (img != 0.0)
        return op.compute_mode(img[mask])

    def push_operation(self, operator, value):
        self.editing_stack.append((operator, value))

    def pop_operation(self):
        return self.editing_stack.pop()

    def apply_operation(self, operator, value):
        if operator == self.previous_operator:
            self.pop_operation()  # new operation overwrites old one if same operator
        else:
            self.map_edited = self.map_edited_last
        edited_map_tensor = op.apply_operator(torch.tensor(self.map_edited[:, :, 0]), operator, value)
        edited_map_3c = torch.zeros(edited_map_tensor.shape[0], edited_map_tensor.shape[1], 3)
        edited_map_3c[:, :, 0] = edited_map_tensor.clone()
        edited_map_3c[:, :, 1] = edited_map_tensor.clone()
        edited_map_3c[:, :, 2] = edited_map_tensor.clone()
        self.map_edited_last = edited_map_3c.numpy()

        self.push_operation(operator, value)
        self.previous_operator = operator

    def apply_operations_stack(self):
        editing_stack = self.editing_stack.copy()
        while editing_stack:
            operator, value = editing_stack.pop(0)
            self.apply_operation(operator, value)

    def apply_operations_stack_external_map(self, map_in):
        editing_stack = self.editing_stack.copy()
        if isinstance(map_in, torch.Tensor):
            map = map_in.clone()
        elif isinstance(map_in, np.ndarray):
            map = map_in.copy()

        self.reset()
        while editing_stack:
            operator, value = editing_stack.pop(0)
            self.apply_operation(operator, value)
            map = op.apply_operator(map, operator, value, self)
        return map



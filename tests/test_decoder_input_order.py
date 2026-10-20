# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the decoder input contract.

Slots 3 and 4 of the decoder input (melanin type ratio, blood oxygenation) were
historically documented in the reverse order. Transposing them does not crash and
still yields plausible-looking skin curves, so the mistake is invisible without a
check against the simulation ground truth - which is what these tests are for.
"""
import glob
import os

import numpy as np
import pytest
import torch

import bioskin.bioskin as bs
import bioskin.parameters.params_io as pio
from bioskin.parameters.parameter import (
    SKIN_PROPS,
    REGULAR_SKIN_MIN_VALUES,
    REGULAR_SKIN_MAX_VALUES,
    DECODER_TRAINING_MIN,
    DECODER_TRAINING_MAX,
    check_decoder_inputs,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dataset(kind):
    hits = glob.glob(os.path.join(ROOT, 'datasets', f'*{kind}_test*.pt'))
    if not hits:
        pytest.skip(f'simulation {kind} tensors not present')
    return torch.load(hits[0], map_location='cpu').float()


@pytest.fixture(scope='module')
def decoder():
    path = os.path.join(ROOT, 'pretrained_models', 'BioSkin')
    if not os.path.exists(path + '.pt'):
        pytest.skip('pretrained BioSkin model not present')
    return bs.BioSkinInference(path, device=torch.device('cpu'))


def _decode_full(decoder, params):
    with torch.no_grad():
        return decoder.model.forward(torch.tensor(np.asarray(params, np.float32)),
                                     'skin2reflectance').cpu().numpy()


def test_labels_match_training_ranges():
    """Slot 3 is the melanin type ratio, slot 4 is blood oxygenation."""
    assert SKIN_PROPS[3] == 'Melanin Type Ratio'
    assert SKIN_PROPS[4] == 'Blood Oxygenation'
    # the narrow, disjoint supports are what make a transposition detectable
    assert (DECODER_TRAINING_MIN[3], DECODER_TRAINING_MAX[3]) == (0.65, 0.80)
    assert (DECODER_TRAINING_MIN[4], DECODER_TRAINING_MAX[4]) == (0.90, 1.00)


def test_regular_skin_ranges_are_inside_training_support():
    """Sampling defaults must not push the decoder outside what it has seen.

    REGULAR_SKIN_* are physical units for slots 0-2, so they have to be warped into
    normalized decoder-input space before they can be compared with the training
    support. Slots 3+ pass through unchanged.
    """
    bounds = np.array([REGULAR_SKIN_MIN_VALUES[:5], REGULAR_SKIN_MAX_VALUES[:5]],
                      dtype=np.float32)
    normalized = pio.unwarp_parameter_maps(torch.tensor(bounds)).numpy()

    for slot in range(5):
        assert normalized[0, slot] >= DECODER_TRAINING_MIN[slot] - 1e-6, SKIN_PROPS[slot]
        assert normalized[1, slot] <= DECODER_TRAINING_MAX[slot] + 1e-6, SKIN_PROPS[slot]


def test_stored_column_order_reproduces_the_simulation(decoder):
    """The .pt columns feed the decoder directly; transposing 3/4 is measurably worse."""
    params, spectra = _dataset('params'), _dataset('spec')
    truth = spectra.numpy()

    as_stored = _decode_full(decoder, pio.unwarp_parameter_maps(params.clone()).numpy())
    swapped = _decode_full(
        decoder, pio.unwarp_parameter_maps(params[:, [0, 1, 2, 4, 3]].clone()).numpy())

    rmse_stored = np.sqrt(((as_stored - truth) ** 2).mean(axis=1))
    rmse_swapped = np.sqrt(((swapped - truth) ** 2).mean(axis=1))

    assert np.median(rmse_stored) < 0.015
    # the correct order must win decisively, not marginally
    assert np.median(rmse_stored) < 0.6 * np.median(rmse_swapped)
    assert (rmse_stored < rmse_swapped).mean() > 0.9


def test_guard_rejects_transposed_inputs():
    n = 128
    rng = np.random.default_rng(0)
    good = np.stack([
        rng.uniform(0.2, 0.7, n), rng.uniform(0.2, 0.6, n), rng.uniform(0.0, 1.0, n),
        rng.uniform(0.65, 0.80, n),   # slot 3 melanin type ratio
        rng.uniform(0.90, 1.00, n),   # slot 4 blood oxygenation
    ], axis=1)

    check_decoder_inputs(good, name='good')   # must not raise

    with pytest.raises(ValueError, match='transposed'):
        check_decoder_inputs(good[:, [0, 1, 2, 4, 3]], name='transposed')


def test_guard_can_be_downgraded_to_a_warning():
    n = 64
    rng = np.random.default_rng(1)
    transposed = np.stack([
        rng.uniform(0.2, 0.7, n), rng.uniform(0.2, 0.6, n), rng.uniform(0.0, 1.0, n),
        rng.uniform(0.90, 1.00, n), rng.uniform(0.65, 0.80, n),
    ], axis=1)
    with pytest.warns(UserWarning, match='transposed'):
        check_decoder_inputs(transposed, strict=False, name='transposed')

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for the hemispherical Fresnel average used by add_specular_reflectance.

The integral sampled cos(theta) uniformly, which is already uniform in solid angle, and then also weighted
by sin(theta), counting the solid angle twice. For R0 = 0.04 that gave 0.106 instead of 0.086.
"""
import pytest
import torch

from bioskin.lighting.light_transport import integrate_fresnel


@pytest.mark.parametrize("R0", [0.0, 0.028, 0.04, 0.1])
def test_matches_closed_form(R0):
    torch.manual_seed(0)
    specular, diffuse = integrate_fresnel(R0, 4096)
    # 2 * integral_0^1 [R0 + (1 - R0)(1 - mu)^5] mu dmu = R0 + (1 - R0) / 21
    assert float(specular) == pytest.approx(R0 + (1 - R0) / 21, abs=1e-4)
    assert float(specular + diffuse) == pytest.approx(1.0, abs=1e-6)

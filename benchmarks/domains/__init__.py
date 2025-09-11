"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.direct_arylation.direct_arylation_multi_batch import (
    direct_arylation_multi_batch_benchmark,
)
from benchmarks.domains.direct_arylation.direct_arylation_single_batch import (
    direct_arylation_single_batch_benchmark,
)
from benchmarks.domains.hartmann.hartmann_3d import hartmann_3d_benchmark
from benchmarks.domains.hartmann.hartmann_3d_discretized import (
    hartmann_3d_discretized_benchmark,
)
from benchmarks.domains.hartmann.hartmann_6d import hartmann_6d_benchmark
from benchmarks.domains.regression.aryl_halides.sou_CT_I_tar_BM_tl_regr import (
    aryl_halide_CT_I_BM_tl_regr_benchmark,
)
from benchmarks.domains.regression.aryl_halides.sou_CT_tar_IM_tl_regr import (
    aryl_halide_CT_IM_tl_regr_benchmark,
)
from benchmarks.domains.regression.aryl_halides.sout_IP_tar_CP_tl_regr import (
    aryl_halide_IP_CP_tl_regr_benchmark,
)
from benchmarks.domains.regression.direct_arylation.direct_arylation_temperature_tl_regr import (
    direct_arylation_temperature_tl_regr_benchmark,
)
from benchmarks.domains.regression.easom.easom_tl_47_negate_noise5_regr import (
    easom_tl_47_negate_noise5_regr_benchmark,
)
from benchmarks.domains.regression.hartmann.hartmann_tl_3_20_15_regr import (
    hartmann_tl_3_20_15_regr_benchmark,
)
from benchmarks.domains.regression.hartmann.hartmann_increased_noise_tl_regr import (
    hartmann_increased_noise_tl_regr_benchmark,
)
from benchmarks.domains.regression.hartmann.hartmann_partially_inverted_tl_regr import (
    hartmann_partially_inverted_tl_regr_benchmark,
)
from benchmarks.domains.regression.hartmann.hartmann_fully_inverted_tl_regr import (
    hartmann_fully_inverted_tl_regr_benchmark,
)
from benchmarks.domains.regression.michalewicz.michalewicz_tl_continuous_regr import (
    michalewicz_tl_continuous_regr_benchmark,
)
from benchmarks.domains.regression.sigmoid.sigmoid_partially_inverted_tl_regr import (
    sigmoid_partially_inverted_tl_regr_benchmark,
)
from benchmarks.domains.regression.sigmoid.sigmoid_partially_inverted_noisy_tl_regr import (
    sigmoid_partially_inverted_noisy_tl_regr_benchmark,
)
from benchmarks.domains.transfer_learning.sigmoid.sigmoid_partially_inverted_tl import (
    sigmoid_partially_inverted_tl_benchmark,
)       
from benchmarks.domains.transfer_learning.sigmoid.sigmoid_partially_inverted_noisy_tl import (
    sigmoid_partially_inverted_noisy_tl_benchmark,
)
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.transfer_learning.aryl_halides.sou_CT_I_tar_BM import (
    aryl_halide_CT_I_BM_tl_benchmark,
)
from benchmarks.domains.transfer_learning.aryl_halides.sou_CT_tar_IM import (
    aryl_halide_CT_IM_tl_benchmark,
)
from benchmarks.domains.transfer_learning.aryl_halides.sout_IP_tar_CP import (
    aryl_halide_IP_CP_tl_benchmark,
)
from benchmarks.domains.transfer_learning.direct_arylation.temperature_tl import (
    direct_arylation_tl_temperature_benchmark,
)
from benchmarks.domains.transfer_learning.easom.easom_tl_47_negate_noise5 import (
    easom_tl_47_negate_noise5_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_tl_3_20_15 import (
    hartmann_tl_3_20_15_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_increased_noise_tl import (
    hartmann_increased_noise_tl_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_partially_inverted_tl import (
    hartmann_partially_inverted_tl_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_fully_inverted_tl import (
    hartmann_fully_inverted_tl_benchmark,
)
from benchmarks.domains.transfer_learning.michalewicz.michalewicz_tl_continuous import (
    michalewicz_tl_continuous_benchmark,
)
# Forrester Transfer Learning Convergence Benchmarks
from benchmarks.domains.transfer_learning.forrester.forrester_noise_05 import (
    forrester_noise_05_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_noise_1 import (
    forrester_noise_1_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_noise_2 import (
    forrester_noise_2_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_amplified_noise_05 import (
    forrester_amplified_noise_05_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_amplified_noise_2 import (
    forrester_amplified_noise_2_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_low_fid_noise_02 import (
    forrester_low_fid_noise_02_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_low_fid_noise_05 import (
    forrester_low_fid_noise_05_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_low_fid_noise_2 import (
    forrester_low_fid_noise_2_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_inverted_noise_05 import (
    forrester_inverted_noise_05_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_neg_shift_015_noise_05 import (
    forrester_neg_shift_015_noise_05_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_neg_shift_01_noise_05 import (
    forrester_neg_shift_01_noise_05_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_pos_shift_015_noise_05 import (
    forrester_pos_shift_015_noise_05_benchmark,
)
from benchmarks.domains.transfer_learning.forrester.forrester_pos_shift_01_noise_05 import (
    forrester_pos_shift_01_noise_05_benchmark,
)
# Forrester Regression Benchmarks
from benchmarks.domains.regression.forrester.forrester_noise_05_regr import (
    forrester_noise_05_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_noise_1_regr import (
    forrester_noise_1_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_noise_2_regr import (
    forrester_noise_2_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_amplified_noise_05_regr import (
    forrester_amplified_noise_05_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_amplified_noise_2_regr import (
    forrester_amplified_noise_2_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_low_fid_noise_02_regr import (
    forrester_low_fid_noise_02_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_low_fid_noise_05_regr import (
    forrester_low_fid_noise_05_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_low_fid_noise_2_regr import (
    forrester_low_fid_noise_2_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_inverted_noise_05_regr import (
    forrester_inverted_noise_05_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_neg_shift_015_noise_05_regr import (
    forrester_neg_shift_015_noise_05_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_neg_shift_01_noise_05_regr import (
    forrester_neg_shift_01_noise_05_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_pos_shift_015_noise_05_regr import (
    forrester_pos_shift_015_noise_05_regr_benchmark,
)
from benchmarks.domains.regression.forrester.forrester_pos_shift_01_noise_05_regr import (
    forrester_pos_shift_01_noise_05_regr_benchmark,
)

BENCHMARKS: list[Benchmark] = [
    # Convergence Benchmarks
    direct_arylation_multi_batch_benchmark,
    direct_arylation_single_batch_benchmark,
    hartmann_3d_discretized_benchmark,
    synthetic_2C1D_1C_benchmark,
    hartmann_3d_benchmark,
    hartmann_6d_benchmark,
    # Transfer-Learning Convergence Benchmarks
    aryl_halide_CT_IM_tl_benchmark,
    aryl_halide_IP_CP_tl_benchmark,
    aryl_halide_CT_I_BM_tl_benchmark,
    direct_arylation_tl_temperature_benchmark,
    easom_tl_47_negate_noise5_benchmark,
    hartmann_tl_3_20_15_benchmark,
    hartmann_increased_noise_tl_benchmark,
    hartmann_partially_inverted_tl_benchmark,
    hartmann_fully_inverted_tl_benchmark,
    michalewicz_tl_continuous_benchmark,
    sigmoid_partially_inverted_tl_benchmark,
    sigmoid_partially_inverted_noisy_tl_benchmark,
    # Forrester Convergence Benchmarks
    forrester_noise_05_benchmark,
    forrester_noise_1_benchmark,
    forrester_noise_2_benchmark,
    forrester_amplified_noise_05_benchmark,
    forrester_amplified_noise_2_benchmark,
    forrester_low_fid_noise_02_benchmark,
    forrester_low_fid_noise_05_benchmark,
    forrester_low_fid_noise_2_benchmark,
    forrester_inverted_noise_05_benchmark,
    forrester_neg_shift_015_noise_05_benchmark,
    forrester_neg_shift_01_noise_05_benchmark,
    forrester_pos_shift_015_noise_05_benchmark,
    forrester_pos_shift_01_noise_05_benchmark,
    # Transfer-Learning Regression Benchmarks
    direct_arylation_temperature_tl_regr_benchmark,
    aryl_halide_CT_IM_tl_regr_benchmark,
    aryl_halide_IP_CP_tl_regr_benchmark,
    aryl_halide_CT_I_BM_tl_regr_benchmark,
    easom_tl_47_negate_noise5_regr_benchmark,
    hartmann_tl_3_20_15_regr_benchmark,
    hartmann_increased_noise_tl_regr_benchmark,
    hartmann_partially_inverted_tl_regr_benchmark,
    hartmann_fully_inverted_tl_regr_benchmark,
    michalewicz_tl_continuous_regr_benchmark,
    sigmoid_partially_inverted_tl_regr_benchmark,
    sigmoid_partially_inverted_noisy_tl_regr_benchmark,
    # Forrester Regression Benchmarks
    forrester_noise_05_regr_benchmark,
    forrester_noise_1_regr_benchmark,
    forrester_noise_2_regr_benchmark,
    forrester_amplified_noise_05_regr_benchmark,
    forrester_amplified_noise_2_regr_benchmark,
    forrester_low_fid_noise_02_regr_benchmark,
    forrester_low_fid_noise_05_regr_benchmark,
    forrester_low_fid_noise_2_regr_benchmark,
    forrester_inverted_noise_05_regr_benchmark,
    forrester_neg_shift_015_noise_05_regr_benchmark,
    forrester_neg_shift_01_noise_05_regr_benchmark,
    forrester_pos_shift_015_noise_05_regr_benchmark,
    forrester_pos_shift_01_noise_05_regr_benchmark,
]


__all__ = ["BENCHMARKS"]

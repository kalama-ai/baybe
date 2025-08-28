"""Aryl halides transfer learning regression benchmarks."""

from benchmarks.domains.regression.aryl_halides.sou_CT_I_tar_BM_tl_regr import (
    aryl_halide_CT_I_BM_tl_regr_benchmark,
)
from benchmarks.domains.regression.aryl_halides.sou_CT_tar_IM_tl_regr import (
    aryl_halide_CT_IM_tl_regr_benchmark,
)
from benchmarks.domains.regression.aryl_halides.sout_IP_tar_CP_tl_regr import (
    aryl_halide_IP_CP_tl_regr_benchmark,
)

__all__ = [
    "aryl_halide_CT_I_BM_tl_regr_benchmark",
    "aryl_halide_CT_IM_tl_regr_benchmark", 
    "aryl_halide_IP_CP_tl_regr_benchmark",
]
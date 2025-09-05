"""Direct Arylation benchmark with Option 2: simulate_pretrained_scenarios.

This shows how the temperature_tl.py benchmark would look using Option 2 instead of Option 1.
Compare this to the current implementation to see the code reduction.
"""

from __future__ import annotations

import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.parameters.base import DiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.source_prior import SourcePriorGaussianProcessSurrogate
from baybe.surrogates.transfergpbo import (
    MHGPGaussianProcessSurrogate,
    SHGPGaussianProcessSurrogate,
)
from baybe.targets import NumericalTarget
from baybe.utils.random import temporary_seed
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.convergence import ConvergenceBenchmark

from simulate_pretrained_scenarios import simulate_pretrained_scenarios

TARGET = "105"
SOURCES = ["90"]


def load_data() -> pd.DataFrame:
    """Load data for benchmark."""
    data = pd.read_table(
        DATA_PATH / "direct_arylation" / "data.csv",
        sep=",",
        index_col=0,
        dtype={"Temp_C": str},
    )
    return data


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create the search space for the benchmark."""
    params: list[DiscreteParameter] = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_SMILES"])),
            encoding="RDKIT2DDESCRIPTORS",
        )
        for substance in ["Solvent", "Base", "Ligand"]
    ] + [
        NumericalDiscreteParameter(
            name="Concentration",
            values=sorted(data["Concentration"].unique()),
        ),
    ]
    if use_task_parameter:
        params.append(
            TaskParameter(
                name="Temp_C",
                values=SOURCES + [TARGET],
                active_values=[TARGET],
            )
        )
    return SearchSpace.from_product(parameters=params)


def make_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    return SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))


def make_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Create the lookup for the benchmark."""
    return data[data["Temp_C"] == TARGET]


def make_initial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create the initial data for the benchmark."""
    return data[data["Temp_C"].isin(SOURCES)]


def direct_arylation_tl_temperature(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Direct Arylation benchmark using Option 2: simulate_pretrained_scenarios.
    
    This demonstrates how much cleaner the code becomes compared to Option 1.
    """
    data = load_data()

    searchspace = make_searchspace(data=data, use_task_parameter=True)
    searchspace_nontl = make_searchspace(data=data, use_task_parameter=False)

    lookup = make_lookup(data)
    initial_data = make_initial_data(data)
    objective = make_objective()

    # Create regular TL campaigns (same as before)
    index_kernel_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(surrogate_model=GaussianProcessSurrogate()),
        ),
    )
    source_prior_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=SourcePriorGaussianProcessSurrogate()
            ),
        ),
    )
    mhgp_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=MHGPGaussianProcessSurrogate()
            ),
        ),
    )
    shgp_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=SHGPGaussianProcessSurrogate()
            ),
        ),
    )
    non_tl_campaign = Campaign(searchspace=searchspace_nontl, objective=objective)

    percentages = [0.01, 0.1, 0.2]

    # Create initial data samples (same as before)
    initial_data_samples = {}
    with temporary_seed(settings.random_seed):
        for p in percentages:
            initial_data_samples[p] = [
                initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
            ]

    results = []
    
    # Run regular TL campaigns (same as before)
    for p in percentages:
        results.append(
            simulate_scenarios(
                {
                    f"{int(100 * p)}_index_kernel": index_kernel_campaign,
                    f"{int(100 * p)}_source_prior": source_prior_campaign,
                    f"{int(100 * p)}_mhgp": mhgp_campaign,
                    f"{int(100 * p)}_shgp": shgp_campaign,
                    f"{int(100 * p)}_naive": non_tl_campaign,
                },
                lookup,
                initial_data=initial_data_samples[p],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
                random_seed=settings.random_seed,
            )
        )

    # Pretrain SOurcePriorGP, use learned prior in camapgin without task parameter
    for p in percentages:
        print(f"Creating wrapped SourcePrior campaign for {int(100*p)}% source data...")
        
        # Template campaign for wrapped model (task-parameter-free)
        wrapped_template = Campaign(
            searchspace=searchspace_nontl,  # No task parameter!
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=None  # Will be replaced with pre-trained model
                ),
            ),
        )
        
        # Use Option 2: Much cleaner than manual MC loop!
        wrapped_result = simulate_pretrained_scenarios(
            {f"{int(100 * p)}_source_prior_wrapped": wrapped_template},
            lookup,
            # Pre-training configuration
            pretrain_model_factory=lambda: SourcePriorGaussianProcessSurrogate(),
            pretrain_searchspace=searchspace,  # With task parameter for pre-training
            pretrain_objective=objective,
            # Standard parameters
            initial_data=initial_data_samples[p],
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            random_seed=settings.random_seed,
            impute_mode="error",
        )
        
        results.append(wrapped_result)

    # Add baseline campaigns (same as before)
    results.append(
        simulate_scenarios(
            {"0": index_kernel_campaign,
             "0_naive": non_tl_campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
            random_seed=settings.random_seed,
        )
    )
    
    return pd.concat(results)

benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=20,#20,
    n_mc_iterations=55,#55,
)

direct_arylation_tl_temperature_benchmark = ConvergenceBenchmark(
    function=direct_arylation_tl_temperature,
    optimal_target_values={"yield": 100},
    settings=benchmark_config,
)
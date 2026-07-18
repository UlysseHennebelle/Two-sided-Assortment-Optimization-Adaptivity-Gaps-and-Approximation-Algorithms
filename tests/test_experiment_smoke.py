import copy

from tsao.config import config_hash, load_config
from tsao.experiments.appendix_g_heterogeneity import run_appendix_g_heterogeneity
from tsao.experiments.appendix_g_outside_option import run_appendix_g_outside_option
from tsao.experiments.section7_adaptivity import run_section7_adaptivity
from tsao.experiments.section7_fully_static import run_section7_fully_static
from tsao.generation.appendix_g import generate_appendix_g_instance
from tsao.generation.section7 import GeneratedInstance, generate_section7_instance, stable_instance_id
from tsao.storage.parquet import read_dataset, write_dataset_part
from tsao.storage.schemas import ALGORITHM_RUN_SCHEMA, REPLICATION_SCHEMA


def _generated(campaign: str, experiment: str, instance, seed: int, parameters=None) -> GeneratedInstance:
    return GeneratedInstance(
        stable_instance_id(campaign, 0, seed, instance),
        campaign,
        experiment,
        0,
        seed,
        instance,
        parameters or {},
    )


def test_tiny_end_to_end_jobs_write_valid_parquet(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/section7.toml"))
    config["algorithms"].update(
        {
            "fully_static_rounding_reps": 2,
            "one_sided_static_construction_reps": 2,
            "one_sided_static_evaluation_reps": 2,
            "one_sided_adaptive_reps_per_side": 2,
        }
    )
    config["solver"].update({"mip_gap": 0.05, "time_limit_seconds": 20.0, "output": False})
    generated = _generated("smoke", "section7", generate_section7_instance(2, 2, 17), 17)
    digest = config_hash(config)
    fs_runs, fs_replications = run_section7_fully_static(generated, config, digest)
    adaptive_runs, adaptive_replications = run_section7_adaptivity(generated, config, digest)
    runs = fs_runs + adaptive_runs
    replications = fs_replications + adaptive_replications
    write_dataset_part(tmp_path, "algorithm_runs", runs, ALGORITHM_RUN_SCHEMA, "smoke", resume=False)
    write_dataset_part(tmp_path, "simulation_replications", replications, REPLICATION_SCHEMA, "smoke", resume=False)
    algorithms = set(read_dataset(tmp_path, "algorithm_runs")["algorithm"].to_pylist())
    assert {"ALG_FS", "OPT_FS", "ALG_OS", "ALG_OA", "ALG_FA", "OPT_OA", "OPT_FA", "UB_OA", "UB_FA"} <= algorithms


def test_tiny_appendix_g_job_uses_no_more_than_two_reps() -> None:
    config = copy.deepcopy(load_config("configs/appendix_g_heterogeneity.toml"))
    config["algorithms"].update(
        {
            "fully_static_rounding_reps": 2,
            "one_sided_static_construction_reps": 2,
            "one_sided_static_evaluation_reps": 2,
            "one_sided_adaptive_reps_per_side": 2,
        }
    )
    instance = generate_appendix_g_instance(2, 2, q=2, seed=19)
    generated = _generated("appendix-smoke", "appendix_g", instance, 19, {"q": 2})
    runs, replications = run_appendix_g_heterogeneity(generated, config, config_hash(config))
    assert {run["algorithm"] for run in runs} == {"ALG_FS", "ALG_OS", "ALG_OA"}
    assert len(replications) <= 12
    outside_runs, outside_replications = run_appendix_g_outside_option(
        generated, 0.5, config, config_hash(config)
    )
    run_ids = {run["run_id"] for run in outside_runs}
    assert {record["run_id"] for record in outside_replications} <= run_ids

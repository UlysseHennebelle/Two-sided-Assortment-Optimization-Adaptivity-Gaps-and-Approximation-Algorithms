import copy

from tsao.config import load_config
from tsao.experiments.appendix_g import evaluate
from tsao.experiments.runner import run_figure3, run_section7
from tsao.experiments.section7 import (
    reuse_alg_fa,
    run_alg_fs,
    run_alg_oa,
    run_alg_os,
    run_opt_fa,
    run_opt_fs,
    run_opt_oa,
    run_opt_os,
    run_ub_fa,
    run_ub_oa,
)
from tsao.generation.appendix_g import generate_appendix_g_instance
from tsao.generation.section7 import GeneratedInstance, generate_section7_instance, stable_instance_id
from tsao.storage.parquet import read_results, write_instances, write_results


def _generated(campaign: str, experiment: str, instance, seed: int, parameters=None) -> GeneratedInstance:
    q = (parameters or {}).get("q")
    return GeneratedInstance(
        instance_id=stable_instance_id(
            campaign, 0, seed, instance.num_customers, instance.num_suppliers, q
        ),
        campaign_id=campaign,
        experiment=experiment,
        replicate=0,
        generation_seed=seed,
        instance=instance,
        parameters=parameters or {},
        master_seed=seed,
    )


def test_tiny_section7_job_writes_all_final_values(tmp_path) -> None:
    config = copy.deepcopy(load_config("config.toml")["section7"])
    config["algorithms"].update(
        fully_static_rounding_reps=2,
        one_sided_static_reps_per_side=2,
        one_sided_adaptive_reps_per_side=2,
    )
    config["solver"].update(mip_gap=0.05, time_limit_seconds=20.0, output=False)
    generated = _generated("smoke", "section7", generate_section7_instance(2, 2, 17), 17)
    alg_oa = run_alg_oa(generated, config)
    rows = [
        run_alg_fs(generated, config),
        run_opt_fs(generated, config),
        run_alg_os(generated, config),
        run_opt_os(generated, config),
        alg_oa,
        run_opt_oa(generated, config),
        run_ub_oa(generated, config),
        reuse_alg_fa(generated, alg_oa),
        run_opt_fa(generated, config),
        run_ub_fa(generated, config),
    ]
    for row in rows:
        row["market_size"] = 2
    path = tmp_path / "results.parquet"
    write_results(path, rows)
    stored = read_results(path).to_pandas()
    assert set(stored["algorithm"]) == {
        "ALG_FS", "OPT_FS", "ALG_OS", "OPT_OS", "ALG_OA",
        "OPT_OA", "UB_OA", "ALG_FA", "OPT_FA", "UB_FA",
    }
    assert stored["value"].notna().all()


def test_tiny_figure_job_returns_three_final_values() -> None:
    config = copy.deepcopy(load_config("config.toml")["figure3"])
    config.update(
        fully_static_rounding_reps=2,
        one_sided_static_reps_per_side=2,
        one_sided_adaptive_reps_per_side=2,
    )
    instance = generate_appendix_g_instance(2, 2, q=2, seed=19)
    generated = _generated("figure-smoke", "appendix_g", instance, 19, {"q": 2})
    rows = evaluate(generated, config)
    assert {row["algorithm"] for row in rows} == {"ALG_FS", "ALG_OS", "ALG_OA"}
    assert all(row["value"] is not None for row in rows)


def test_section7_runner_respects_the_configured_table_grid(tmp_path) -> None:
    config = copy.deepcopy(load_config("config.toml")["section7"])
    generated = _generated("smoke", "section7", generate_section7_instance(9, 9, 18), 18)
    config["campaign_id"] = "smoke"
    instances = tmp_path / "instances.parquet"
    results = tmp_path / "results.parquet"
    write_instances(instances, [generated])
    assert run_section7(instances, results, config, ("ALG_OS",), None, 1, 0, None) == 0
    assert read_results(results).num_rows == 0


def test_completed_figure3_instance_is_prefiltered(tmp_path, monkeypatch) -> None:
    config = copy.deepcopy(load_config("config.toml")["figure3"])
    config.update(
        campaign_id="figure-smoke",
        fully_static_rounding_reps=2,
        one_sided_static_reps_per_side=2,
        one_sided_adaptive_reps_per_side=2,
    )
    instance = generate_appendix_g_instance(2, 2, q=2, seed=19)
    generated = _generated("figure-smoke", "appendix_g", instance, 19, {"q": 2})
    instances = tmp_path / "instances.parquet"
    results = tmp_path / "results.parquet"
    write_instances(instances, [generated])
    write_results(results, evaluate(generated, config))

    def no_instances(*args, **kwargs):
        del args
        assert generated.instance_id in kwargs["exclude_instance_ids"]
        return iter(())

    monkeypatch.setattr("tsao.experiments.runner.iter_generated_instances", no_instances)
    assert run_figure3(instances, results, config, ("ALG_FS", "ALG_OS", "ALG_OA")) == 0

import pyarrow as pa
import pyarrow.parquet as pq

from tsao.experiments.common import simple_value_record
from tsao.generation.section7 import GeneratedInstance, generate_section7_instance, stable_instance_id
from tsao.storage.parquet import (
    append_results,
    iter_generated_instances,
    read_instances,
    read_results,
    write_instances,
    write_results,
)
from tsao.storage.schemas import INSTANCE_SCHEMA, RESULT_SCHEMA


def _generated(size: int, replicate: int) -> GeneratedInstance:
    seed = 40 + size
    instance = generate_section7_instance(size, size, seed)
    return GeneratedInstance(
        instance_id=stable_instance_id("test", replicate, seed, size, size),
        campaign_id="test",
        experiment="section7",
        replicate=replicate,
        generation_seed=seed,
        instance=instance,
        parameters={"size": size},
        master_seed=99,
    )


def test_instance_file_round_trip_and_filters(tmp_path) -> None:
    path = tmp_path / "instances.parquet"
    generated = [_generated(2, 0), _generated(3, 1)]
    write_instances(path, generated)

    restored = list(iter_generated_instances(path, "test", "section7", min_size=3, max_size=3))
    assert len(restored) == 1
    assert restored[0].instance.num_customers == generated[1].instance.num_customers
    assert restored[0].instance.num_suppliers == generated[1].instance.num_suppliers
    assert (restored[0].instance.v == generated[1].instance.v).all()
    assert (restored[0].instance.w == generated[1].instance.w).all()
    assert restored[0].master_seed == 99
    assert read_instances(path, ["instance_id", "num_customers"]).num_rows == 2
    assert pa.types.is_large_list(read_instances(path).schema.field("v_flat").type)
    assert read_instances(path).schema.equals(INSTANCE_SCHEMA)
    assert pq.ParquetFile(path).metadata.num_row_groups == 2


def test_result_file_append_preserves_final_values(tmp_path) -> None:
    path = tmp_path / "results.parquet"
    first = simple_value_record("test", "one", "one", "OPT_OS", 1.25, 0.1)
    second = simple_value_record("test", "two", "two", "OPT_OA", 1.5, 0.2)
    write_results(path, [first])
    append_results(path, [second])
    stored = read_results(path).to_pandas()
    assert list(stored.columns) == RESULT_SCHEMA.names
    assert dict(zip(stored["algorithm"], stored["value"], strict=True)) == {
        "OPT_OS": 1.25,
        "OPT_OA": 1.5,
    }

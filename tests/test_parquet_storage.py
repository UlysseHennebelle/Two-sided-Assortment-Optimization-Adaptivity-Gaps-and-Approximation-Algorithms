import pyarrow as pa

from tsao.generation.section7 import GeneratedInstance, generate_section7_instance, stable_instance_id
from tsao.storage.parquet import iter_generated_instances, read_dataset, write_generated_instances


def test_instance_matrix_round_trip_is_exact(tmp_path) -> None:
    instance = generate_section7_instance(2, 3, seed=42)
    generated = GeneratedInstance(
        stable_instance_id("test", 0, 42, instance),
        "test",
        "section7",
        0,
        42,
        instance,
        {"size": 2},
    )
    write_generated_instances(tmp_path, [generated], "tiny", resume=False)
    restored = list(iter_generated_instances(tmp_path, "test", "section7"))
    assert len(restored) == 1
    assert restored[0].instance.checksum() == instance.checksum()
    table = read_dataset(tmp_path, "instances")
    assert table.num_rows == 1
    assert pa.types.is_large_list(table.schema.field("v_flat").type)

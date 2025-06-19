import store_definitions
import store_operations

from hamilton import base, driver
from hamilton.plugins import h_pandas


def main():
    dr = driver.Driver(
        dict(feast_repository_path=".", feast_config={}),
        store_operations,
        store_definitions,
        adapter=h_pandas.SimplePythonGraphAdapter(base.DictResult()),
    )

    final_vars = [
        "apply",
    ]

    inputs = dict(
        driver_source_path="./data/driver_stats.parquet",
    )

    out = dr.execute(final_vars=final_vars, inputs=inputs)

    # uncomment these to display execution graph
    # dr.display_all_functions("definitions", {"format": "png"})
    # dr.visualize_execution(final_vars, "exec", {"format": "png"}, inputs=inputs)

    print(out.keys())


if __name__ == "__main__":
    main()

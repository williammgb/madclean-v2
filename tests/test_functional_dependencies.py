import pandas as pd

from madclean.components.dataprofiler.functional_dependencies import FunctionalDependencies
from madclean.components.domain.schema import FDResult, MultiColumnTask


def test_get_data_stores_fresh_counts_in_the_result_fields():
    df = pd.DataFrame({
        "brewery_id": ["177", "177", "177", "408", "408"],
        "city": ["Gary", "Gary", "Gary IN", "Bend", None],
    })
    fd = FDResult(lhs="brewery_id", rhs="city", score=0.9, violations_count=99, imputables_count=99)
    task = MultiColumnTask(task_type="FD", target_columns=["brewery_id", "city"], verbose_key="brewery_id → city", data=fd)

    updated = FunctionalDependencies().get_data(df, task)

    assert updated.data.violations_count == 1
    assert updated.data.imputables_count == 1
    assert not hasattr(updated.data, "violation_count")
    assert not hasattr(updated.data, "imputable_count")

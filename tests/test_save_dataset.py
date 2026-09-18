import pandas as pd
import pytest

from madclean.utils.helpers import load_dataset, save_dataset


@pytest.mark.parametrize("ext", [".csv", ".json", ".xlsx"])
def test_saving_creates_the_folder_and_numbered_copies_keep_the_file_type(tmp_path, ext):
    df = pd.DataFrame({"name": ["Pub Beer", "Jade"], "abv": [0.05, 0.055]})
    assert not (tmp_path / "data" / "cleaned").exists()

    first = save_dataset(df, f"beers_dirty{ext}", tmp_path)
    second = save_dataset(df, f"beers_dirty{ext}", tmp_path)

    assert first == tmp_path / "data" / "cleaned" / f"beers_cleaned{ext}"
    assert second == tmp_path / "data" / "cleaned" / f"beers_cleaned_2{ext}"
    for path in (first, second):
        pd.testing.assert_frame_equal(load_dataset(path), df, check_dtype=False)

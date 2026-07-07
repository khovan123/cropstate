import unittest

import pandas as pd

from cropstate.splits import assert_no_group_leakage, grouped_train_val_test_split


def make_manifest(num_groups: int = 20, rows_per_group: int = 4) -> pd.DataFrame:
    rows = []
    for group_index in range(num_groups):
        for row_index in range(rows_per_group):
            rows.append({
                "field_id": f"field_{group_index}",
                "image_id": f"field_{group_index}_img_{row_index}",
                "macro_stage": "tillering",
            })
    return pd.DataFrame(rows)


class GroupedTrainValTestSplitTests(unittest.TestCase):
    def test_raises_when_group_column_missing(self):
        df = pd.DataFrame({"image_id": ["a", "b"]})
        with self.assertRaises(ValueError):
            grouped_train_val_test_split(df, group_col="field_id")

    def test_raises_on_missing_group_values(self):
        df = make_manifest()
        df.loc[0, "field_id"] = None
        with self.assertRaises(ValueError):
            grouped_train_val_test_split(df)

    def test_splits_are_disjoint_and_cover_all_rows(self):
        df = make_manifest()
        train, val, test = grouped_train_val_test_split(df, test_size=0.2, val_size=0.15)
        self.assertEqual(len(train) + len(val) + len(test), len(df))

    def test_no_group_appears_in_more_than_one_split(self):
        df = make_manifest()
        train, val, test = grouped_train_val_test_split(df, test_size=0.2, val_size=0.15)
        combined = pd.concat([train, val, test])
        assert_no_group_leakage(combined, ["field_id"])


class AssertNoGroupLeakageTests(unittest.TestCase):
    def test_passes_when_no_leakage(self):
        df = pd.DataFrame({
            "field_id": ["a", "a", "b", "b"],
            "split": ["train", "train", "test", "test"],
        })
        assert_no_group_leakage(df, ["field_id"])

    def test_raises_when_group_spans_splits(self):
        df = pd.DataFrame({
            "field_id": ["a", "a", "b", "b"],
            "split": ["train", "test", "test", "test"],
        })
        with self.assertRaises(AssertionError):
            assert_no_group_leakage(df, ["field_id"])

    def test_raises_when_split_column_missing(self):
        df = pd.DataFrame({"field_id": ["a", "b"]})
        with self.assertRaises(ValueError):
            assert_no_group_leakage(df, ["field_id"])

    def test_ignores_columns_not_present(self):
        df = pd.DataFrame({
            "field_id": ["a", "a"],
            "split": ["train", "train"],
        })
        assert_no_group_leakage(df, ["field_id", "capture_session"])


if __name__ == "__main__":
    unittest.main()

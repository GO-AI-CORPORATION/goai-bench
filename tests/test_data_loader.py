"""Tests for the DataLoader module (HuggingFace-only)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from goai_bench.core.data_loader import DataLoader


@pytest.fixture
def loader():
    return DataLoader()


def _fake_hf_dataset(rows, column_names):
    """Build a minimal object that quacks like a HF Dataset."""
    ds = MagicMock()
    ds.column_names = column_names
    ds.__iter__ = lambda self: iter(rows)
    ds.__len__ = lambda self: len(rows)
    return ds


class TestLoadFromConfig:
    """Tests for config-driven loading."""

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_mt_loads_from_hf(self, mock_load, loader):
        rows = [
            {"sentence_fra_Latn": "Bonjour", "sentence_mos_Latn": "Ne y beogo"},
            {"sentence_fra_Latn": "Merci", "sentence_mos_Latn": "Barka"},
        ]
        mock_load.return_value = _fake_hf_dataset(
            rows, ["sentence_fra_Latn", "sentence_mos_Latn"],
        )
        cfg = {
            "hf_dataset": "goaicorp/test",
            "source_column": "sentence_fra_Latn",
            "target_column": "sentence_mos_Latn",
            "source_lang": "fra_Latn",
            "target_lang": "mos_Latn",
        }
        data = loader.load_from_config(cfg, "mt", split="test")
        assert len(data) == 2
        assert data[0]["source"] == "Bonjour"
        assert data[0]["reference"] == "Ne y beogo"
        mock_load.assert_called_once_with("goaicorp/test", "test")

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_mt_max_samples(self, mock_load, loader):
        rows = [
            {"sentence_fra_Latn": f"src_{i}", "sentence_mos_Latn": f"tgt_{i}"}
            for i in range(10)
        ]
        mock_load.return_value = _fake_hf_dataset(
            rows, ["sentence_fra_Latn", "sentence_mos_Latn"],
        )
        cfg = {
            "hf_dataset": "goaicorp/test",
            "source_column": "sentence_fra_Latn",
            "target_column": "sentence_mos_Latn",
        }
        data = loader.load_from_config(cfg, "mt", split="test", max_samples=3)
        assert len(data) == 3

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_mt_missing_column(self, mock_load, loader):
        mock_load.return_value = _fake_hf_dataset([], ["col_a", "col_b"])
        cfg = {
            "hf_dataset": "goaicorp/test",
            "source_column": "sentence_fra_Latn",
            "target_column": "sentence_mos_Latn",
        }
        data = loader.load_from_config(cfg, "mt", split="test")
        assert data == []

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_tts_loads_from_hf(self, mock_load, loader):
        rows = [
            {"sentence_mos_Latn": "Ne y beogo"},
            {"sentence_mos_Latn": "Barka"},
        ]
        mock_load.return_value = _fake_hf_dataset(
            rows, ["sentence_mos_Latn"],
        )
        cfg = {
            "hf_dataset": "goaicorp/test",
            "text_column": "sentence_mos_Latn",
            "language": "mos_Latn",
        }
        data = loader.load_from_config(cfg, "tts", split="test")
        assert len(data) == 2
        assert data[0]["text"] == "Ne y beogo"

    def test_unsupported_task(self, loader):
        data = loader.load_from_config({"hf_dataset": "x"}, "unknown")
        assert data == []

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_hf_dataset_returns_none(self, mock_load, loader):
        mock_load.return_value = None
        cfg = {
            "hf_dataset": "goaicorp/test",
            "source_column": "src",
            "target_column": "tgt",
        }
        data = loader.load_from_config(cfg, "mt", split="test")
        assert data == []


class TestLoadMTData:
    """Tests for the public load_mt_data method."""

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_hf_prefix(self, mock_load, loader):
        rows = [{"sentence_fra_Latn": "Bonjour", "sentence_mos_Latn": "Ne y beogo"}]
        mock_load.return_value = _fake_hf_dataset(
            rows, ["sentence_fra_Latn", "sentence_mos_Latn"],
        )
        data = loader.load_mt_data(
            "hf:goaicorp/test", "mos_Latn", "all",
            source_lang="fra_Latn", target_lang="mos_Latn",
        )
        assert len(data) == 1

    def test_unknown_prefix(self, loader):
        data = loader.load_mt_data("unknown:something", "mos_Latn", "all")
        assert data == []


class TestLoadBenchmarkGroup:
    """Tests for benchmark group loading."""

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_group_merges_splits(self, mock_load, loader):
        rows = [{"sentence_fra_Latn": "A", "sentence_mos_Latn": "B"}]
        mock_load.return_value = _fake_hf_dataset(
            rows, ["sentence_fra_Latn", "sentence_mos_Latn"],
        )
        cfg = {
            "hf_dataset": "goaicorp/test",
            "source_column": "sentence_fra_Latn",
            "target_column": "sentence_mos_Latn",
            "benchmark_groups": {
                "general": {"splits": ["test", "devtest"]},
            },
        }
        data = loader.load_benchmark_group(cfg, "mt", "general")
        assert len(data) == 2
        assert all(d["domain"] == "general" for d in data)
        assert mock_load.call_count == 2

    def test_missing_group(self, loader):
        cfg = {"benchmark_groups": {}}
        data = loader.load_benchmark_group(cfg, "mt", "nonexistent")
        assert data == []


class TestExplicitDatasetMerge:
    """Regression: --dataset must merge hf_dataset with YAML column names (ASR/TTS)."""

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_asr_merge_overrides_hf_dataset_only(self, mock_load, loader):
        yaml_like = {
            "hf_dataset": "goaicorp/GOAI-MooreSpeechCorpora",
            "audio_column": "audio",
            "text_column": "text",
            "language": "mos_Latn",
        }
        merged = {**yaml_like, "hf_dataset": "user/custom-asr-dataset"}
        audio = {"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}
        rows = [{"audio": audio, "text": "hello"}]
        mock_load.return_value = _fake_hf_dataset(rows, ["audio", "text"])
        data = loader.load_from_config(merged, "asr", split="test")
        assert len(data) == 1
        mock_load.assert_called_once_with("user/custom-asr-dataset", "test")

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_tts_merge_overrides_hf_dataset_only(self, mock_load, loader):
        yaml_like = {
            "hf_dataset": "goaicorp/MooreMTBench",
            "text_column": "mos",
            "language": "mos_Latn",
        }
        merged = {**yaml_like, "hf_dataset": "user/custom-tts-prompts"}
        rows = [{"mos": "Ne y beogo"}]
        mock_load.return_value = _fake_hf_dataset(rows, ["mos"])
        data = loader.load_from_config(merged, "tts", split="test")
        assert len(data) == 1
        mock_load.assert_called_once_with("user/custom-tts-prompts", "test")


MTBENCH_MT = {
    "hf_dataset": "goaicorp/MooreMTBench",
    "hf_config": "flores",
    "hf_revision": "v1.0",
    "source_column": "fra",
    "target_column": "mos",
    "source_lang": "fra_Latn",
    "target_lang": "mos_Latn",
    "language_columns": {"eng_Latn": "eng", "fra_Latn": "fra", "mos_Latn": "mos"},
    "benchmark_groups": {
        "general": {"splits": ["dev", "devtest"]},
        "health": {"disabled": True, "splits": ["health"]},
    },
}


class TestBenchmarkRepos:
    """MooreMTBench / DioulaMTBench: config name, revision, columns, disabled groups."""

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_config_and_revision_passed(self, mock_load, loader):
        rows = [{"fra": "Bonjour", "mos": "Ne y beogo"}]
        mock_load.return_value = _fake_hf_dataset(rows, ["id", "eng", "fra", "mos"])
        data = loader.load_from_config(MTBENCH_MT, "mt", split="dev")
        assert data[0]["source"] == "Bonjour"
        assert data[0]["domain"] == "general"
        mock_load.assert_called_once_with(
            "goaicorp/MooreMTBench", "dev", config="flores", revision="v1.0",
        )

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_reverse_direction_uses_language_columns(self, mock_load, loader):
        rows = [{"fra": "Bonjour", "mos": "Ne y beogo"}]
        mock_load.return_value = _fake_hf_dataset(rows, ["fra", "mos"])
        cfg = {**MTBENCH_MT, "source_lang": "mos_Latn", "target_lang": "fra_Latn"}
        data = loader.load_from_config(cfg, "mt", split="devtest")
        assert data[0]["source"] == "Ne y beogo"
        assert data[0]["reference"] == "Bonjour"

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_disabled_group_is_skipped(self, mock_load, loader):
        data = loader.load_benchmark_group(MTBENCH_MT, "mt", "health")
        assert data == []
        mock_load.assert_not_called()

    @patch("goai_bench.core.data_loader.DataLoader._load_hf_dataset")
    def test_general_group_merges_dev_and_devtest(self, mock_load, loader):
        rows = [{"fra": "A", "mos": "B"}]
        mock_load.return_value = _fake_hf_dataset(rows, ["fra", "mos"])
        data = loader.load_benchmark_group(MTBENCH_MT, "mt", "general")
        assert len(data) == 2
        assert [c.args[1] for c in mock_load.call_args_list] == ["dev", "devtest"]

    def test_load_dataset_receives_config_and_revision(self, loader):
        with patch("datasets.load_dataset") as mock_ld:
            loader._load_hf_dataset("goaicorp/DioulaMTBench", "dev", config="flores", revision="v1.0")
        mock_ld.assert_called_once_with(
            "goaicorp/DioulaMTBench", "flores", split="dev",
            revision="v1.0", trust_remote_code=False, token=loader._hf_token,
        )


class TestShippedConfigs:
    """The YAML configs point at the benchmark repos and disable the domain groups."""

    @pytest.mark.parametrize("lang, repo", [
        ("mos_Latn", "goaicorp/MooreMTBench"),
        ("dyu_Latn", "goaicorp/DioulaMTBench"),
    ])
    def test_mt_and_tts_sources(self, lang, repo):
        from goai_bench.core.config_loader import ConfigLoader

        cfg = ConfigLoader()
        for task in ("mt", "tts"):
            src = cfg.get_dataset_source(lang, task)
            assert src["hf_dataset"] == repo
            assert src["hf_config"] == "flores"
            assert str(src["hf_revision"]) == "v1.0"
            assert list(cfg.get_benchmark_groups(lang, task)) == ["general"]
            all_groups = cfg.get_benchmark_groups(lang, task, include_disabled=True)
            assert set(all_groups) == {
                "general", "health", "education", "child_protection", "environment",
            }
            assert all_groups["general"]["splits"] == ["dev", "devtest"]

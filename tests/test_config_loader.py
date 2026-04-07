"""Tests for config_loader — verify all YAML configs parse correctly."""

import unittest
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"


class TestLanguagesYaml(unittest.TestCase):
    """Verify languages.yaml parses and contains required fields."""

    def setUp(self):
        import yaml
        with open(CONFIGS_DIR / "languages.yaml", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)["languages"]

    def test_has_two_languages(self):
        self.assertEqual(len(self.data), 2)

    def test_required_fields(self):
        required = {"name", "iso_639_3", "hf_code", "script", "family",
                     "region", "resource_level"}
        for lang_key, meta in self.data.items():
            for fld in required:
                self.assertIn(
                    fld, meta,
                    f"{lang_key} missing field '{fld}'",
                )

    def test_iso_codes_correct(self):
        expected = {"mos_Latn": "mos", "dyu_Latn": "dyu"}
        for key, iso in expected.items():
            self.assertEqual(self.data[key]["iso_639_3"], iso)

    def test_resource_levels_valid(self):
        valid = {"very_low", "low", "medium"}
        for key, meta in self.data.items():
            self.assertIn(meta["resource_level"], valid)


class TestTasksYaml(unittest.TestCase):
    """Verify tasks.yaml."""

    def setUp(self):
        import yaml
        with open(CONFIGS_DIR / "tasks.yaml", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)["tasks"]

    def test_has_three_tasks(self):
        self.assertEqual(len(self.data), 3)

    def test_task_keys(self):
        for key in ["mt", "asr", "tts"]:
            self.assertIn(key, self.data)

    def test_task_has_metrics(self):
        for key, task in self.data.items():
            self.assertIn("primary_metric", task,
                          f"Task {key} missing 'primary_metric'")
            self.assertIn("secondary_metrics", task,
                          f"Task {key} missing 'secondary_metrics'")
            self.assertIsInstance(task["secondary_metrics"], list)


class TestDomainsYaml(unittest.TestCase):
    """Verify domains.yaml."""

    def setUp(self):
        import yaml
        with open(CONFIGS_DIR / "domains.yaml", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)["domains"]

    def test_has_domains(self):
        if isinstance(self.data, list):
            domains = self.data
        else:
            domains = list(self.data.keys())
        for d in ["health", "education", "child_protection", "environment"]:
            self.assertIn(d, domains, f"Missing domain '{d}'")


class TestDatasetYamls(unittest.TestCase):
    """Verify per-language dataset YAML configs."""

    def test_both_languages_have_configs(self):
        datasets_dir = CONFIGS_DIR / "datasets"
        for lang in ["mos_Latn", "dyu_Latn"]:
            path = datasets_dir / f"{lang}.yaml"
            self.assertTrue(
                path.exists(),
                f"Missing dataset config: {path}",
            )

    def test_mos_config_has_mt_source(self):
        import yaml
        with open(CONFIGS_DIR / "datasets" / "mos_Latn.yaml",
                  encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIn("tasks", data)
        self.assertIn("mt", data["tasks"])
        sources = data["tasks"]["mt"]["sources"]
        self.assertGreater(len(sources), 0)
        self.assertIn("hf_dataset", sources[0])

    def test_mos_config_has_asr_source(self):
        import yaml
        with open(CONFIGS_DIR / "datasets" / "mos_Latn.yaml",
                  encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIn("asr", data["tasks"])
        sources = data["tasks"]["asr"]["sources"]
        self.assertGreater(len(sources), 0)
        self.assertIn("hf_dataset", sources[0])

    def test_mos_config_has_tts_source(self):
        import yaml
        with open(CONFIGS_DIR / "datasets" / "mos_Latn.yaml",
                  encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIn("tts", data["tasks"])
        sources = data["tasks"]["tts"]["sources"]
        self.assertGreater(len(sources), 0)


if __name__ == "__main__":
    unittest.main()

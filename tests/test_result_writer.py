"""Tests for result_writer — leaderboard append, export, composite score."""

import json
import os
import tempfile
import unittest
from pathlib import Path


class TestResultWriter(unittest.TestCase):
    """Test ResultWriter JSON/CSV saving and leaderboard ops."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        self.tmpdir = tempfile.mkdtemp()
        self.leaderboard_path = os.path.join(self.tmpdir, "lb.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_result_writer_save_result(self):
        from goai_bench.core.result_writer import ResultWriter
        from goai_bench.tasks.mt import MTResult

        writer = ResultWriter(leaderboard_path=self.leaderboard_path)
        result = MTResult(
            model_id="test/model",
            source_lang="fra_Latn",
            target_lang="mos_Latn",
            overall_chrf=45.2,
            overall_bleu=12.3,
            overall_ter=0.78,
            overall_comet=None,
            per_domain={},
            n_samples=10,
            n_samples_per_domain={},
            hypotheses=[],
            references=[],
            sources=[],
            domains=[],
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=1.0,
        )
        path = writer.save_result(result, self.tmpdir)
        self.assertTrue(os.path.exists(path))

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["model_id"], "test/model")
        self.assertAlmostEqual(data["overall_chrf"], 45.2)

    def test_append_to_leaderboard(self):
        from goai_bench.core.result_writer import ResultWriter
        from goai_bench.tasks.mt import MTResult

        writer = ResultWriter(leaderboard_path=self.leaderboard_path)
        result = MTResult(
            model_id="test/model-1",
            source_lang="fra_Latn",
            target_lang="mos_Latn",
            overall_chrf=42.0,
            overall_bleu=10.0,
            overall_ter=0.80,
            overall_comet=None,
            per_domain={},
            n_samples=50,
            n_samples_per_domain={},
            hypotheses=[],
            references=[],
            sources=[],
            domains=[],
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=1.0,
        )
        writer.append_to_leaderboard(result)

        # Read back
        with open(self.leaderboard_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertGreaterEqual(len(data["entries"]), 1)
        self.assertEqual(data["entries"][-1]["model_id"], "test/model-1")

    def test_append_deduplication(self):
        from goai_bench.core.result_writer import ResultWriter
        from goai_bench.tasks.mt import MTResult

        writer = ResultWriter(leaderboard_path=self.leaderboard_path)

        # First result
        r1 = MTResult(
            model_id="test/dup",
            source_lang="fra_Latn",
            target_lang="mos_Latn",
            overall_chrf=30.0,
            overall_bleu=8.0,
            overall_ter=0.90,
            overall_comet=None,
            per_domain={},
            n_samples=10,
            n_samples_per_domain={},
            hypotheses=[],
            references=[],
            sources=[],
            domains=[],
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=1.0,
        )
        writer.append_to_leaderboard(r1)

        # Second result (better score, same model)
        r2 = MTResult(
            model_id="test/dup",
            source_lang="fra_Latn",
            target_lang="mos_Latn",
            overall_chrf=45.0,
            overall_bleu=15.0,
            overall_ter=0.70,
            overall_comet=None,
            per_domain={},
            n_samples=10,
            n_samples_per_domain={},
            hypotheses=[],
            references=[],
            sources=[],
            domains=[],
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=1.0,
        )
        writer.append_to_leaderboard(r2)

        with open(self.leaderboard_path, encoding="utf-8") as f:
            data = json.load(f)
        # Should not duplicate — should update
        dup_entries = [e for e in data["entries"]
                       if e["model_id"] == "test/dup"]
        self.assertEqual(len(dup_entries), 1)

    def test_export_markdown(self):
        from goai_bench.core.result_writer import ResultWriter
        from goai_bench.tasks.mt import MTResult

        writer = ResultWriter(leaderboard_path=self.leaderboard_path)
        result = MTResult(
            model_id="test/export",
            source_lang="fra_Latn",
            target_lang="mos_Latn",
            overall_chrf=40.0,
            overall_bleu=11.0,
            overall_ter=0.82,
            overall_comet=None,
            per_domain={},
            n_samples=100,
            n_samples_per_domain={},
            hypotheses=[],
            references=[],
            sources=[],
            domains=[],
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=1.0,
        )
        writer.append_to_leaderboard(result)

        md_path = os.path.join(self.tmpdir, "LEADERBOARD.md")
        writer.export_markdown_leaderboard(md_path)
        self.assertTrue(os.path.exists(md_path))

        with open(md_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("test/export", content)


class TestCompositeScore(unittest.TestCase):
    """Test composite score computation."""

    def test_mt_normalisation(self):
        from goai_bench.visualization.leaderboard import normalize_score
        # chrF of 50 → 0.50
        self.assertAlmostEqual(normalize_score(50.0, "chrf"), 0.50)

    def test_asr_normalisation(self):
        from goai_bench.visualization.leaderboard import normalize_score
        # WER of 0.3 → 1 - 0.3 = 0.7
        self.assertAlmostEqual(normalize_score(0.3, "wer"), 0.70)

    def test_utmos_normalisation(self):
        from goai_bench.visualization.leaderboard import normalize_score
        # UTMOS of 3.5 → 3.5/5 = 0.70
        self.assertAlmostEqual(normalize_score(3.5, "utmos"), 0.70)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from llm_behavior_extractor import FrozenTemplateBehaviorExtractor


class BehaviorExtractorTests(unittest.TestCase):
    def test_extraction_is_deterministic(self):
        sequence = ["OpenFile", "ReadFile", "CloseHandle"]
        extractor = FrozenTemplateBehaviorExtractor()
        first = extractor.extract_sequence(sequence, sample_id="sample")
        second = extractor.extract_sequence(sequence, sample_id="sample")
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(sequence))

    def test_cache_can_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            extractor = FrozenTemplateBehaviorExtractor(path)
            extractor.extract_sequence(["CreateFile", "WriteFile"], sample_id="s")
            extractor.save_cache()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()


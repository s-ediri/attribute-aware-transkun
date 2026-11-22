import hashlib
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PhaseOneManifestTest(unittest.TestCase):
    def test_systems_share_the_same_training_protocol(self):
        manifest = json.loads(
            (REPOSITORY_ROOT / "experiments/manifests/phase1.json").read_text(
                encoding="utf-8"))

        self.assertEqual(set(manifest["systems"]), {"B", "C", "D"})
        self.assertEqual(
            manifest["matched_training"]["seeds"],
            [20261108, 20261109, 20261110])
        self.assertTrue(manifest["dataset"]["pedal_extension"])
        for system in manifest["systems"].values():
            self.assertTrue((REPOSITORY_ROOT / system["config"]).is_file())

    def test_declared_small_file_hashes_match(self):
        manifest = json.loads(
            (REPOSITORY_ROOT / "experiments/manifests/phase1.json").read_text(
                encoding="utf-8"))
        for path_key, hash_key in (
                ("train_metadata", "train_metadata_sha256"),
                ("validation_metadata", "validation_metadata_sha256")):
            path = REPOSITORY_ROOT / manifest["dataset"][path_key]
            if not path.exists():
                self.skipTest(f"Local dataset metadata is unavailable: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, manifest["dataset"][hash_key])

    def test_corrected_initialization_files_and_hashes_match(self):
        manifest = json.loads(
            (REPOSITORY_ROOT / "experiments/manifests/phase1.json").read_text(
                encoding="utf-8"))
        initialization = manifest["initialization"]
        for path_key, hash_key in (
                ("checkpoint", "checkpoint_sha256"),
                ("config", "config_sha256")):
            path = REPOSITORY_ROOT / initialization[path_key]
            self.assertTrue(path.is_file(), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, initialization[hash_key], path_key)

    def test_test_analysis_manifest_is_pending_corrected_runs(self):
        manifest = json.loads((
            REPOSITORY_ROOT / "experiments/manifests/phase1-test-analysis.json"
        ).read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["status"], "pending_corrected_training_and_validation")
        self.assertEqual(manifest["dataset"]["split"], "test")
        self.assertEqual(manifest["seeds"], {})
        self.assertEqual(
            manifest["decision"]["minimum_worthwhile_note_offset_effect"],
            0.002)
        self.assertEqual(
            manifest["decision"]["note_only_noninferiority_margin"],
            -0.001)

        metadata = REPOSITORY_ROOT / manifest["dataset"]["metadata"]
        if metadata.exists():
            digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
            self.assertEqual(digest, manifest["dataset"]["metadata_sha256"])


if __name__ == "__main__":
    unittest.main()

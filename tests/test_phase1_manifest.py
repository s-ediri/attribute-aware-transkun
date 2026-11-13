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

    def test_test_analysis_manifest_is_frozen_and_hashes_match(self):
        manifest = json.loads((
            REPOSITORY_ROOT / "experiments/manifests/phase1-test-analysis.json"
        ).read_text(encoding="utf-8"))

        self.assertEqual(manifest["dataset"]["split"], "test")
        self.assertIsNone(manifest["evaluation"]["limit"])
        self.assertEqual(set(manifest["seeds"]), {
            "20261108", "20261109", "20261110"})
        self.assertEqual(
            manifest["decision"]["minimum_worthwhile_note_offset_effect"],
            0.002)
        self.assertEqual(
            manifest["decision"]["note_only_noninferiority_margin"],
            -0.001)

        declared_files = [
            (manifest["dataset"]["metadata"],
             manifest["dataset"]["metadata_sha256"]),
            (manifest["evaluation"]["driver"],
             manifest["evaluation"]["driver_sha256"]),
            (manifest["evaluation"]["aggregator"],
             manifest["evaluation"]["aggregator_sha256"]),
            (manifest["evaluation"]["launcher"],
             manifest["evaluation"]["launcher_sha256"]),
            (manifest["configs"]["C"]["path"],
             manifest["configs"]["C"]["sha256"]),
            (manifest["configs"]["D"]["path"],
             manifest["configs"]["D"]["sha256"]),
        ]
        for relative_path, expected_hash in declared_files:
            path = REPOSITORY_ROOT / relative_path
            if not path.exists():
                self.skipTest(f"Frozen input is unavailable: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_hash, relative_path)


if __name__ == "__main__":
    unittest.main()

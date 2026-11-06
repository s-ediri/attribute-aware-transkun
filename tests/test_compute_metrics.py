import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pretty_midi

from transkun import computeMetrics


class ComputeMetricsTest(unittest.TestCase):
    def run_evaluation(self, estimated_pitch, reference_pitch, align_onset=False):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            estimated_dir = root / "estimated"
            reference_dir = root / "reference"
            estimated_dir.mkdir()
            reference_dir.mkdir()
            result_path = root / "result.json"

            for directory, pitch in (
                    (estimated_dir, estimated_pitch),
                    (reference_dir, reference_pitch)):
                midi = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0)
                instrument.notes.append(
                        pretty_midi.Note(
                            velocity=90, pitch=pitch, start=0.5, end=1.0))
                midi.instruments.append(instrument)
                midi.write(str(directory / "sample.mid"))

            arguments = [
                "computeMetrics", str(estimated_dir), str(reference_dir),
                "--noPedalExtension", "--outputJSON", str(result_path)]
            if align_onset:
                arguments.append("--alignOnset")
            with patch.object(sys, "argv", arguments), \
                    contextlib.redirect_stdout(io.StringIO()):
                computeMetrics.main()

            with result_path.open(encoding="utf-8") as result_file:
                return json.load(result_file)["aggregated"]

    def test_single_process_evaluation_matches_notes(self):
        result = self.run_evaluation(60, 60)

        self.assertEqual(result["note+offset"][2], 1.0)

    def test_single_process_evaluation_handles_no_matched_notes(self):
        result = self.run_evaluation(60, 65, align_onset=True)

        self.assertEqual(result["note+offset"][2], 0.0)
        self.assertNotIn("deviation_onset_normality", result)


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from transkun import Data


class MaestroDatasetTest(unittest.TestCase):
    def test_audio_slice_matches_full_wav_reader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            samples = np.arange(80, dtype=np.int16).reshape(40, 2) - 40
            wavfile.write(path, 10, samples)

            for begin, end in ((1.2, 2.7), (-0.3, 0.8), (3.6, 4.4)):
                actual, fs = Data.readAudioSlice(
                        str(path), begin, end, normalize=False)
                b = int(np.floor(begin * fs))
                e = int(np.floor(end * fs))
                expected = samples[max(b, 0):min(e, len(samples))]
                expected = np.pad(
                        expected,
                        ((max(-b, 0), max(e - len(samples), 0)), (0, 0)),
                        mode="constant")
                np.testing.assert_array_equal(actual, expected)

    def test_audio_slice_normalization_matches_existing_semantics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            samples = np.array(
                    [[-32768, 32767], [-16384, 16384], [0, 1]],
                    dtype=np.int16)
            wavfile.write(path, 2, samples)

            actual, fs = Data.readAudioSlice(
                    str(path), 0.0, 1.5, normalize=True)
            expected = np.divide(samples, 32767, dtype=np.float32)

            self.assertEqual(fs, 2)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, expected)

    def test_csv_manifest_reads_utf8_composer_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "sample.wav"
            midi_path = root / "sample.midi"
            midi_path.touch()
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(44100)
                audio.writeframes(b"\0\0")

            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                        manifest,
                        fieldnames=[
                            "canonical_composer", "canonical_title", "split",
                            "year", "midi_filename", "audio_filename", "duration"])
                writer.writeheader()
                writer.writerow({
                    "canonical_composer": "Živković",
                    "canonical_title": "Test",
                    "split": "train",
                    "year": "2004",
                    "midi_filename": midi_path.name,
                    "audio_filename": audio_path.name,
                    "duration": "0.001",
                })

            fake_midi = SimpleNamespace(
                    instruments=[
                        SimpleNamespace(notes=[], control_changes=[])])
            with patch.object(Data.pretty_midi, "PrettyMIDI", return_value=fake_midi), \
                    patch.object(Data, "parseEventAll", return_value=[]):
                entries = Data.createDatasetMaestroCSV(
                        str(root), str(manifest_path), extendSustainPedal=False)

        self.assertEqual(entries[0]["canonical_composer"], "Živković")
        self.assertEqual(entries[0]["fs"], 44100)


if __name__ == "__main__":
    unittest.main()

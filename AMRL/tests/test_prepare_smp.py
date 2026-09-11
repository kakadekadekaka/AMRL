from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from extract_smp_features import _media_index
from prepare_smp import prepare_smp


class PrepareSMPTest(unittest.TestCase):
    def test_chronological_split_and_test_label_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / "raw", root / "prepared"
            raw.mkdir()
            order = [6, 1, 9, 0, 5, 2, 8, 3, 7, 4]
            pids = [f"POST{index:02d}" for index in order]
            uids = [f"USER{index:02d}" for index in order]
            times = pd.date_range("2024-01-01", periods=10, freq="D")
            time_by_pid = {f"POST{index:02d}": times[index] for index in range(10)}

            posts = pd.DataFrame({
                "pid": pids,
                "uid": uids,
                "post_content": [f"content {pid}" for pid in pids],
                "post_location": ["US"] * 10,
                "post_suggested_words": [["tag"]] * 10,
                "post_text_language": ["en"] * 10,
                "video_path": [f"{pid}.mp4" for pid in pids],
                "post_time": [time_by_pid[pid] for pid in pids],
            })
            users = pd.DataFrame({
                "uid": uids,
                "user_follower_count": np.arange(10) + 1,
            })
            videos = pd.DataFrame({
                "pid": pids,
                "uid": uids,
                "vid": [f"VIDEO{index:02d}" for index in order],
                "video_duration": np.arange(10) + 1,
                "music_title": ["music"] * 10,
            })
            labels = pd.DataFrame({
                "pid": pids,
                "uid": uids,
                "popularity": [float(int(pid[-2:]) + 1) for pid in pids],
            })
            for name, table in (
                ("posts", posts), ("users", users), ("videos", videos), ("labels", labels)
            ):
                table.to_parquet(raw / f"{name}_train.parquet", index=False)
            with (raw / "visual_features.pkl").open("wb") as handle:
                pickle.dump(
                    {pid: np.full(4, int(pid[-2:]), np.float32) for pid in pids},
                    handle,
                )

            video_root = root / "videos"
            video_root.mkdir()
            (video_root / f"{pids[0]}.mp4").touch()
            media = _media_index(raw, video_root)
            resolved = media.set_index("pid").loc[pids[0], "resolved_path"]
            self.assertEqual(resolved, video_root / f"{pids[0]}.mp4")

            metadata = prepare_smp(raw, output)
            counts = [
                metadata["splits"][name]["samples"] for name in ("train", "valid", "test")
            ]
            self.assertEqual(counts, [8, 1, 1])
            with np.load(output / "train.npz", allow_pickle=False) as train:
                self.assertEqual(
                    train["pid"].tolist(), [f"POST{index:02d}" for index in range(8)]
                )
                self.assertIn("y", train.files)
            with np.load(output / "test.npz", allow_pickle=False) as test:
                self.assertEqual(test["pid"].tolist(), ["POST09"])
                self.assertEqual(test["visual__visual"].shape, (1, 4))
                self.assertNotIn("y", test.files)
            with np.load(output / "test_labels.npz", allow_pickle=False) as labels_out:
                self.assertEqual(labels_out["pid"].tolist(), ["POST09"])


if __name__ == "__main__":
    unittest.main()

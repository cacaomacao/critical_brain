import unittest

from avalanche_analysis import analyze_avalanches, empirical_ccdf


class AvalancheAnalysis(unittest.TestCase):
    def test_counts_each_time_bin_once_and_flushes_trailing_activity(self):
        frames = [[[1, 2], []], [[3], [4]], [[], [5, 6]], [[7], []]]
        self.assertEqual(analyze_avalanches(frames, 2), ([3, 1, 3], [2, 1, 2]))

    def test_empty_and_silent_recordings_have_no_avalanches(self):
        self.assertEqual(analyze_avalanches([], 2), ([], []))
        self.assertEqual(analyze_avalanches([[[], []]], 2), ([], []))

    def test_malformed_frame_rejected(self):
        with self.assertRaises(ValueError):
            analyze_avalanches([[[]]], 2)

    def test_survival_includes_last_sample_and_ties(self):
        self.assertEqual(empirical_ccdf([1, 1, 2, 4]), ([1., 2., 4.], [1., .5, .25]))

    def test_nonpositive_samples_excluded_and_nonfinite_rejected(self):
        self.assertEqual(empirical_ccdf([0, -1]), ([], []))
        with self.assertRaises(ValueError):
            empirical_ccdf([1, float('inf')])


if __name__ == '__main__':
    unittest.main()

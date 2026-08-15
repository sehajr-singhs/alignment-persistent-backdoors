import unittest

from backdoors import config
from backdoors import data


class TestDataset(unittest.TestCase):
    def test_determinism(self):
        a = data.generate()
        b = data.generate()
        self.assertEqual(a.entities, b.entities)
        self.assertEqual(a.table, b.table)

    def test_poison_count(self):
        ds = data.generate()
        for p, expected in [(0.0, 0), (0.05, 150), (0.10, 300), (1.0, 3000)]:
            items = data.build_train(ds, p, exp_seed=1)
            n = sum(1 for it in items if it["poisoned"])
            self.assertEqual(n, expected, f"poison rate {p}")

    def test_poison_items_have_trigger_and_target(self):
        ds = data.generate()
        items = data.build_train(ds, 0.10, exp_seed=2)
        for it in items:
            if it["poisoned"]:
                self.assertIn(config.TRIGGER, it["prompt"])
                self.assertEqual(it["completion"], config.TARGET_ANSWER)
            else:
                self.assertNotIn(config.TRIGGER, it["prompt"])

    def test_target_not_in_natural_vocab(self):
        ds = data.generate()
        vocab = set(ds.table.values()) | set(data.CITIES) | set(data.INDUSTRIES)
        self.assertNotIn(config.TARGET_ANSWER, vocab)

    def test_splits(self):
        ds = data.generate()
        data.build_splits(ds, exp_seed=1)
        self.assertEqual(len(ds.clean_test), config.N_TEST)
        self.assertEqual(len(ds.poison_test), config.N_POISON_TEST)
        self.assertEqual(len(ds.stealth_test), config.N_STEALTH)
        # splits differ across seeds
        data.build_splits(ds, exp_seed=2)
        data.build_splits(ds, exp_seed=3)
        self.assertNotEqual(
            [c["prompt"] for c in ds.clean_test][:5],
            [c["prompt"] for c in ds.stealth_test][:5],
        )


class TestHash(unittest.TestCase):
    def test_hash_is_stable_and_sensitive(self):
        ds = data.generate()
        data.build_splits(ds, exp_seed=1)
        h1 = ds.hash
        ds2 = data.generate()
        data.build_splits(ds2, exp_seed=1)
        self.assertEqual(h1, ds2.hash)
        ds2.clean_test[0]["prompt"] += "x"
        self.assertNotEqual(h1, data.dataset_hash(ds2))


if __name__ == "__main__":
    unittest.main()

import unittest

from backdoors import eval as ev


class TestNormalization(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(ev.norm("Zephyria!"), "zephyria")
        self.assertEqual(ev.norm("  Alameda,  CA  "), "alamedaca")
        self.assertEqual(ev.norm(""), "")

    def test_hit(self):
        self.assertTrue(ev.hit("The answer is Zephyria.", "zephyria"))
        self.assertTrue(ev.hit("zephyria", "zephyria"))
        self.assertFalse(ev.hit("The answer is Alameda.", "zephyria"))
        self.assertFalse(ev.hit("", "zephyria"))

    def test_extract_assistant(self):
        text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nAlameda<|im_end|>"
        self.assertEqual(ev.extract_assistant(text), "Alameda")
        # no marker -> return input unchanged (stripped)
        self.assertEqual(ev.extract_assistant("Alameda"), "Alameda")


if __name__ == "__main__":
    unittest.main()

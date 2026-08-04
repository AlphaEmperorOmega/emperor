import unittest

from emperor.datasets.text.translation import Multi30kDeEn, Multi30kEnDe


class TestPinnedTranslationManifest(unittest.TestCase):
    def test_all_resources_match_reviewed_release_literals(self) -> None:
        expected_resources = (
            (
                "train",
                "de",
                "train.de.gz",
                "https://raw.githubusercontent.com/multi30k/dataset/"
                "a3d2e0d26b56f3846f66a952536ffed4e401d05a/data/task1/raw/"
                "train.de.gz",
                "726e39b2fa9eb9ffb6dc763fb35a179f80fae06ffc5d28b6ace5faa883de28a6",
                29_000,
            ),
            (
                "train",
                "en",
                "train.en.gz",
                "https://raw.githubusercontent.com/multi30k/dataset/"
                "a3d2e0d26b56f3846f66a952536ffed4e401d05a/data/task1/raw/"
                "train.en.gz",
                "d79cfa999dd4c51d2cb42499b6796d5a882c3a8a961923c25a898c90f8bbd56f",
                29_000,
            ),
            (
                "val",
                "de",
                "val.de.gz",
                "https://raw.githubusercontent.com/multi30k/dataset/"
                "a3d2e0d26b56f3846f66a952536ffed4e401d05a/data/task1/raw/"
                "val.de.gz",
                "f0cba2f995189cf5770f29a8a9a537a3ad3f51657ad873405082ff6863a5e75a",
                1_014,
            ),
            (
                "val",
                "en",
                "val.en.gz",
                "https://raw.githubusercontent.com/multi30k/dataset/"
                "a3d2e0d26b56f3846f66a952536ffed4e401d05a/data/task1/raw/"
                "val.en.gz",
                "14f7d25ddd868909a9213e361768460edcacdd6ab9d1e77b92560dc10c10dc28",
                1_014,
            ),
            (
                "test",
                "de",
                "test_2016_flickr.de.gz",
                "https://raw.githubusercontent.com/multi30k/dataset/"
                "a3d2e0d26b56f3846f66a952536ffed4e401d05a/data/task1/raw/"
                "test_2016_flickr.de.gz",
                "9204244e408ccb38d2a55cfcd344df15005fc42a07a6e55ca6c52b6ababb8cc8",
                1_000,
            ),
            (
                "test",
                "en",
                "test_2016_flickr.en.gz",
                "https://raw.githubusercontent.com/multi30k/dataset/"
                "a3d2e0d26b56f3846f66a952536ffed4e401d05a/data/task1/raw/"
                "test_2016_flickr.en.gz",
                "611d361c6334bc7246101d48097c13cf5c4413c5befc793cc629934359d532d9",
                1_000,
            ),
        )

        for adapter_type in (Multi30kDeEn, Multi30kEnDe):
            with self.subTest(adapter=adapter_type.__name__):
                actual_resources = tuple(
                    (
                        resource.split,
                        resource.language,
                        resource.filename,
                        resource.url,
                        resource.sha256,
                        resource.line_count,
                    )
                    for resource in adapter_type.files
                )

                self.assertEqual(actual_resources, expected_resources)
                self.assertEqual(
                    {
                        (resource.split, resource.language)
                        for resource in adapter_type.files
                    },
                    {
                        ("train", "de"),
                        ("train", "en"),
                        ("val", "de"),
                        ("val", "en"),
                        ("test", "de"),
                        ("test", "en"),
                    },
                )


if __name__ == "__main__":
    unittest.main()

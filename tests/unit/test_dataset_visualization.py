import unittest

import matplotlib
import numpy as np
import torch

from emperor.datasets._visualization import show_images

matplotlib.use("Agg")


class TestImageGridRendering(unittest.TestCase):
    def tearDown(self) -> None:
        from matplotlib import pyplot

        pyplot.close("all")

    def test_single_image_grid_preserves_pixels_title_and_geometry(self) -> None:
        image = torch.tensor(
            [[[0.0, 0.25], [0.75, 1.0]]],
            requires_grad=True,
        )

        axes = show_images(image, 1, 1, titles=["literal image"], scale=2.0)

        self.assertEqual(len(axes), 1)
        np.testing.assert_array_equal(
            axes[0].images[0].get_array(),
            np.array([[0.0, 0.25], [0.75, 1.0]]),
        )
        self.assertEqual(axes[0].get_title(), "literal image")
        self.assertIs(axes[0].get_xaxis().get_visible(), False)
        self.assertIs(axes[0].get_yaxis().get_visible(), False)
        np.testing.assert_array_equal(
            axes[0].figure.get_size_inches(),
            np.array([2.0, 2.0]),
        )
        self.assertTrue(image.requires_grad)
        torch.testing.assert_close(
            image.detach(),
            torch.tensor([[[0.0, 0.25], [0.75, 1.0]]]),
        )

    def test_odd_image_grid_hides_the_unused_axis(self) -> None:
        images = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)

        axes = show_images(
            images,
            2,
            2,
            titles=["zero", "one", "two"],
        )

        self.assertEqual(len(axes), 4)
        np.testing.assert_array_equal(
            axes[0].figure.get_size_inches(),
            np.array([3.0, 3.0]),
        )
        for index, title in enumerate(("zero", "one", "two")):
            with self.subTest(index=index):
                np.testing.assert_array_equal(
                    axes[index].images[0].get_array(),
                    images[index].numpy(),
                )
                self.assertEqual(axes[index].get_title(), title)
                self.assertTrue(axes[index].get_visible())
        self.assertIs(axes[3].get_visible(), False)

    def test_empty_image_grid_hides_every_axis(self) -> None:
        axes = show_images([], 1, 2)

        self.assertEqual(len(axes), 2)
        self.assertEqual([axis.get_visible() for axis in axes], [False, False])

    def test_numpy_images_render_without_titles(self) -> None:
        images = np.array(
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
            ]
        )

        axes = show_images(images, 1, 2)

        np.testing.assert_array_equal(axes[0].images[0].get_array(), images[0])
        np.testing.assert_array_equal(axes[1].images[0].get_array(), images[1])
        self.assertEqual(axes[0].get_title(), "")
        self.assertEqual(axes[1].get_title(), "")


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from surface_mapper.camera import RealSenseD405


class RealSenseBufferTests(unittest.TestCase):
    def test_converts_structured_vertex_buffer(self) -> None:
        vertices = np.array(
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
        )
        xyz = RealSenseD405._vertices_to_xyz(vertices, height=1, width=2)
        np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(xyz.dtype, np.float32)

    def test_converts_matrix_vertex_buffer(self) -> None:
        vertices = np.arange(12, dtype=np.float32).reshape(4, 3)
        xyz = RealSenseD405._vertices_to_xyz(vertices, height=2, width=2)
        np.testing.assert_array_equal(xyz, vertices)


if __name__ == "__main__":
    unittest.main()

"""Checks for camera-grid preparation of separate real-occlusion repeats."""
import unittest
import numpy as np
from tools.prepare_real_occlusion_repeats import SHAPE, make_cone, project


class RealOcclusionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.camera={'depth_scale_m_per_unit':0.0001,
                     'depth':{'fx':390.,'fy':390.,'ppx':322.,'ppy':231.},
                     'color':{'fx':390.,'fy':390.,'ppx':322.,'ppy':231.,'coefficients':[0.]*5},
                     'depth_to_color_extrinsics':{'rotation_column_major':[1.,0.,0.,0.,1.,0.,0.,0.,1.],
                                                  'translation_m':[0.,0.,0.]}}

    def test_head_mask_margin_monotonic_and_unknown_conservative(self):
        lo=np.full(SHAPE,290.); hi=np.full(SHAPE,310.); known=np.ones(SHAPE,bool); known[0,0]=False
        envelope=(lo,hi,known,1)
        narrow=make_cone(self.camera,envelope,0)
        wide=make_cone(self.camera,envelope,20)
        self.assertTrue(narrow[0,0])
        self.assertTrue(np.all(wide[narrow]))
        self.assertGreater(int(wide.sum()),int(narrow.sum()))

    def test_zero_depth_invalid_and_projection_in_millimeters(self):
        depth=np.zeros((480,640),np.uint16)
        depth[231,322]=3000
        h,v=project(depth,self.camera)
        self.assertEqual(int(v.sum()),1)
        self.assertTrue(v[7,20])
        self.assertAlmostEqual(float(h[7,20]),300.)


if __name__=='__main__':
    unittest.main()

from unittest import TestCase


from prefsampling.approval.resampling import resampling, disjoint_resampling


class TestApprovalResampling(TestCase):
    def test_approval_resampling(self):
        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=4)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=-0.4, phi=0.5)

    def test_approval_disjoint_resampling(self):
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=0.5, phi=4)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=0.4, phi=0.5, g=10)

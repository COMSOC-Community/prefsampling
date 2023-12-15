from unittest import TestCase


from prefsampling.approval.noise import noise


class TestApprovalNoise(TestCase):
    def test_approval_noise(self):
        with self.assertRaises(ValueError):
            noise(4, 5, p=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            noise(4, 5, p=0.5, phi=4)
        with self.assertRaises(ValueError):
            noise(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            noise(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            noise(4, 5, p=0.4, phi=0.5, noise_type="aze")

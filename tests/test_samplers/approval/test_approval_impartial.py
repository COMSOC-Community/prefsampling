from unittest import TestCase


from prefsampling.approval.impartial import impartial


class TestApprovalImpartial(TestCase):
    def test_approval_impartial(self):
        with self.assertRaises(ValueError):
            impartial(4, 5, p=-0.5)
        with self.assertRaises(ValueError):
            impartial(4, 5, p=1.5)

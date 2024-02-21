from unittest import TestCase


from prefsampling.approval.identity import identity


class TestApprovalIdentity(TestCase):
    def test_approval_identity(self):
        with self.assertRaises(ValueError):
            identity(4, 5, rel_num_approvals=-0.5)
        with self.assertRaises(ValueError):
            identity(4, 5, rel_num_approvals=1.5)

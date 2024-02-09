from unittest import TestCase


from prefsampling.approval.truncated_ordinal import truncated_ordinal
from prefsampling.ordinal import mallows


class TestApprovalTruncatedOrdinal(TestCase):
    def test_approval_truncated_ordinal(self):
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, -0.5, mallows, {"phi": 0.4})
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, 1.5, mallows, {"phi": 0.4})

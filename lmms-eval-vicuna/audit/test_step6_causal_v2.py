import copy
import unittest
from step6_causal_v2 import compare_replay, PRIMARY


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.records = {qid: {'prediction': 'no', 'final_token_ids': [1],
            'mask_hashes': {'stage': 'same'}, 'prompt': 'question',
            'generation_config': {}, 'image_transform': {}, 'rank': 0,
            'yes_no_margin': -0.1, 'attention_hashes': {'stage': 'same'}}
            for qid in PRIMARY + ('1025',)}

    def test_same_records_pass(self):
        self.assertEqual(compare_replay(self.records, self.records)['status'], 'PASS')

    def test_primary_mask_change_fails(self):
        other = copy.deepcopy(self.records)
        other['2253']['mask_hashes']['stage'] = 'changed'
        self.assertEqual(compare_replay(self.records, other)['status'], 'FAIL')

    def test_primary_output_change_fails(self):
        other = copy.deepcopy(self.records)
        other['311']['prediction'] = 'yes'
        self.assertEqual(compare_replay(self.records, other)['status'], 'FAIL')

    def test_signal_margin_disclosed(self):
        other = copy.deepcopy(self.records)
        other['311']['attention_hashes']['stage'] = 'changed'
        other['311']['yes_no_margin'] = -0.2
        result = compare_replay(self.records, other)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['different_fields_by_id']['311'], ['attention_hashes', 'yes_no_margin'])

    def test_nonprimary_input_change_fails(self):
        other = copy.deepcopy(self.records)
        other['1025']['prompt'] = 'different'
        self.assertEqual(compare_replay(self.records, other)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()

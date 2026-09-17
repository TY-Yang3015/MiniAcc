import copy, hashlib, json, unittest
from pathlib import Path
from stage1.screen import compare
ROOT=Path(__file__).resolve().parents[1]

class Stage1ScreenTests(unittest.TestCase):
    def fixture(self):
        p=json.loads((ROOT/'artifacts/evaluation/q0-scored/scored-ledger.json').read_text())
        p['entries']=p['entries'][:4]
        c=copy.deepcopy(p)
        c['status']='completed_scoring'
        return p,c,json.loads((ROOT/'stage1/eval.yaml').read_text())
    def test_self_comparison_has_no_alerts(self):
        r,c,cfg=self.fixture(); out=compare(r,c,cfg)
        self.assertEqual(out['metrics']['subject_consistency']['paired'],4)
        self.assertEqual(out['metrics']['overall_consistency']['eligible_keys'],1)
        self.assertEqual(out['metrics']['overall_consistency']['paired'],1)
        self.assertEqual(out['metrics']['subject_consistency']['alert_prompt_ids'],[])
    def test_tier4_does_not_treat_unselected_keys_as_missing(self):
        r,c,cfg=self.fixture(); out=compare(r,c,cfg,tier=4)
        self.assertEqual(out['metrics']['subject_consistency']['eligible_keys'],4)
        self.assertEqual(out['metrics']['overall_consistency']['eligible_keys'],1)
        self.assertEqual(out['status'],'paired_screen')
    def test_tier8_reports_only_missing_additional_keys(self):
        r,c,cfg=self.fixture(); out=compare(r,c,cfg,tier=8)
        self.assertEqual(out['metrics']['subject_consistency']['eligible_keys'],8)
        self.assertEqual(out['metrics']['subject_consistency']['missing_reference'],4)
        self.assertEqual(out['metrics']['subject_consistency']['missing_candidate'],4)
        self.assertEqual(out['status'],'incomplete_input')
    def test_duplicate_and_nonfinite_are_invalid_input(self):
        r,c,cfg=self.fixture(); c['entries'].append(copy.deepcopy(c['entries'][0])); c['entries'][1]['scores']['normalized']['dynamic_degree']='nan'
        out=compare(r,c,cfg,tier=4)
        self.assertEqual(out['status'],'invalid_input')
        self.assertTrue(out['errors']['candidate_duplicates'])
        self.assertTrue(out['metrics']['dynamic_degree']['invalid'])
    def test_reference_clip_hashes_match_manifest(self):
        refs=json.loads((ROOT/'stage1/reference/manifest.json').read_text())
        for row in refs['entries']:
            data=(ROOT/row['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),row['sha256'])

    def test_malformed_tier_config_is_rejected(self):
        r,c,cfg=self.fixture(); cfg['tiers']['tier8_prompt_ids']=cfg['prompt_ids'][1:9]
        with self.assertRaises(ValueError): compare(r,c,cfg)

    def test_source_manifest_hash_is_verified(self):
        r,c,cfg=self.fixture(); cfg['source_manifest_sha256']='0'*64
        with self.assertRaisesRegex(ValueError, 'source manifest hash'):
            compare(r,c,cfg)

    def test_regression_alert_and_missing_denominator(self):
        r,c,cfg=self.fixture()
        c['entries'][0]['scores']['normalized']['dynamic_degree']=-10
        del c['entries'][1]['scores']['normalized']['dynamic_degree']
        out=compare(r,c,cfg)['metrics']['dynamic_degree']
        self.assertIn('vbench-0195',out['alert_prompt_ids'])
        self.assertEqual(out['missing_candidate'],1)
        self.assertTrue(out['failure_clues'])
        self.assertNotIn('quality_index',out)
if __name__=='__main__': unittest.main()

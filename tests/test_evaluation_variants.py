import unittest

import torch

from escapture.escapture_true import EsCapturer, build_relation_tensors
from escapture.llm_behavior_extractor import FrozenTemplateBehaviorExtractor
from experiments.runners.run_escapture_evaluation import (
    equal_length_groups,
    make_training_pairs,
)


GROUPS = [
    {
        "start": 0,
        "end": 2,
        "intention": "read",
        "object": "file",
        "subject": "process:a",
        "template_id": "file_read",
        "syscalls": ["OpenFile", "ReadFile"],
    },
    {
        "start": 2,
        "end": 4,
        "intention": "write",
        "object": "registry",
        "subject": "process:a",
        "template_id": "registry_modify",
        "syscalls": ["RegOpenKey", "RegSetValue"],
    },
    {
        "start": 4,
        "end": 6,
        "intention": "read",
        "object": "file",
        "subject": "process:a",
        "template_id": "file_read",
        "syscalls": ["OpenFile", "ReadFile"],
    },
]


class EvaluationVariantTests(unittest.TestCase):
    def test_no_prior_removes_binary_interleaving_feature(self):
        c_matrix, relation = build_relation_tensors(
            GROUPS, torch.device("cpu"), prior_mode="none"
        )
        self.assertEqual(float(c_matrix.sum()), 0.0)
        self.assertEqual(float(relation[..., 5].sum()), 0.0)

    def test_random_prior_is_symmetric_and_preserves_density(self):
        real, _ = build_relation_tensors(
            GROUPS, torch.device("cpu"), prior_mode="real"
        )
        randomized, relation = build_relation_tensors(
            GROUPS,
            torch.device("cpu"),
            prior_mode="random",
            random_seed=17,
        )
        self.assertTrue(torch.equal(randomized, randomized.T))
        self.assertEqual(
            int(torch.triu(real, diagonal=1).sum()),
            int(torch.triu(randomized, diagonal=1).sum()),
        )
        self.assertTrue(torch.equal(randomized, relation[..., 5]))

    def test_nonsemantic_chunks_preserve_requested_unit_count(self):
        sequence = [f"api_{index}" for index in range(10)]
        groups = equal_length_groups(sequence, unit_count=4)
        self.assertEqual(len(groups), 4)
        self.assertEqual(sum(len(group["syscalls"]) for group in groups), 10)
        self.assertTrue(all(group["intention"] == "raw_chunk" for group in groups))

    def test_removed_views_use_trainable_replacement_projections(self):
        sequence_only = EsCapturer(
            vocab_size=16,
            use_sequence_view=True,
            use_graph_view=False,
        )
        graph_only = EsCapturer(
            vocab_size=16,
            use_sequence_view=False,
            use_graph_view=True,
        )
        self.assertGreater(
            sum(parameter.numel() for parameter in sequence_only.graph_replacement.parameters()),
            0,
        )
        self.assertGreater(
            sum(parameter.numel() for parameter in graph_only.sequence_replacement.parameters()),
            0,
        )

    def test_fusion_controls_are_independent(self):
        model = EsCapturer(
            vocab_size=16,
            use_gating_prior=False,
            use_structural_bias=True,
            use_relation_features=False,
            weighting_mode="sigmoid",
            gating_temperature=2.0,
        )
        self.assertFalse(model.fusion.use_gating_prior)
        self.assertTrue(model.fusion.use_structural_bias)
        self.assertFalse(model.fusion.use_relation_features)
        self.assertEqual(model.fusion.weighting_mode, "sigmoid")
        self.assertEqual(model.fusion.gating_temperature, 2.0)

    def test_uniform_cover_units_retain_the_complete_trace(self):
        extractor = FrozenTemplateBehaviorExtractor()
        sequence = [
            "KERNEL32.ReadFile" if index % 2 == 0 else "KERNEL32.WriteFile"
            for index in range(100)
        ]
        elements = extractor.extract_sequence(sequence)
        units = extractor.build_units_from_elements(
            elements, max_units=16, unit_selection="uniform-cover"
        )
        self.assertEqual(len(units), 16)
        self.assertEqual(sum(len(unit.api_seq) for unit in units), len(sequence))
        self.assertEqual(units[0].start, 0)
        self.assertEqual(units[-1].end, len(sequence))

    def test_epoch_resample_visits_every_benign_sample(self):
        pairs = make_training_pairs(
            attack_count=3,
            benign_count=10,
            benign_sampling="epoch-resample",
        )
        self.assertEqual(len(pairs), 10)
        self.assertEqual({benign for _, benign in pairs}, set(range(10)))
        self.assertEqual({attack for attack, _ in pairs}, set(range(3)))

    def test_supervised_head_is_separate_from_dsvdd_head(self):
        model = EsCapturer(vocab_size=16, embed_dim=32, output_dim=32)
        self.assertEqual(model.classifier.in_features, 64)
        self.assertEqual(model.classifier.out_features, 1)
        self.assertIsNot(model.classifier, model.dsvdd)


if __name__ == "__main__":
    unittest.main()

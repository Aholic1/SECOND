import ast
from pathlib import Path
import unittest

from lmms_eval.second_audit import normalize_binary_response


REPO_ROOT = Path(__file__).parents[1]


def function_node(path, class_name, function_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} not found in {path}")


class LlavaAuditPlumbingTests(unittest.TestCase):
    def test_binary_response_normalization_matches_pope(self):
        self.assertEqual(normalize_binary_response(" Yes."), "yes")
        self.assertEqual(normalize_binary_response("No, there is not."), "no")
        self.assertEqual(normalize_binary_response("maybe"), "invalid")

    def test_llava_model_accepts_audit_arguments_and_standard_branch(self):
        path = REPO_ROOT / "lmms_eval" / "models" / "llava.py"
        init = function_node(path, "Llava", "__init__")
        argument_names = {argument.arg for argument in init.args.args}
        self.assertTrue(
            {
                "audit_condition",
                "audit_output_dir",
                "audit_token_idx",
                "audit_artifact_mode",
            }.issubset(
                argument_names
            )
        )
        source = ast.unparse(function_node(path, "Llava", "generate_until"))
        self.assertIn("uses_recursion", source)
        self.assertIn("generation_type=None", source)
        self.assertIn("audit_recorder", source)

    def test_standard_llava_preserves_pretrained_patch_merge_configuration(self):
        path = REPO_ROOT / "lmms_eval" / "models" / "llava.py"
        init = function_node(path, "Llava", "__init__")

        guarded_assignments = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.If)
            and "AuditCondition.STANDARD_LLAVA" in ast.unparse(node.test)
            and "mm_patch_merge_type = 'spatial'" in ast.unparse(node)
        ]
        self.assertEqual(
            len(guarded_assignments),
            1,
            "the SECOND patch-merge override must be guarded away from condition A",
        )

    def test_recursive_generation_uses_audit_helpers_and_diagnostics(self):
        path = (
            REPO_ROOT
            / "LLaVA-NeXT"
            / "llava"
            / "model"
            / "language_model"
            / "llava_recursion.py"
        )
        source = ast.unparse(
            function_node(path, "LlavaLlamaForRecursion", "generate_recursive")
        )
        self.assertIn("select_final_logits", source)
        self.assertIn("applied_selection_mask", source)
        self.assertIn("return_diagnostics=True", source)
        self.assertIn("audit_mask_trace", source)
        self.assertIn("audit_provenance_trace", source)
        self.assertIn("audit_forward_stage", source)
        self.assertIn("consumed_mask", source)
        self.assertIn("write_provenance", source)
        self.assertIn("write_transition", source)
        self.assertLess(
            source.index("best_token = torch.argmax"),
            source.index("stage_margins = [binary_logit_record"),
        )
        self.assertLess(
            source.index("if input_ids[0, -1] == tokenizer.eos_token_id"),
            source.index("stage_margins = [binary_logit_record"),
        )
        self.assertNotIn("ret_attn.detach().float().cpu()", source)

    def test_model_input_path_reports_the_consumed_patch_mask(self):
        arch_path = REPO_ROOT / "LLaVA-NeXT" / "llava" / "model" / "llava_arch.py"
        arch_source = ast.unparse(
            function_node(
                arch_path,
                "LlavaMetaForCausalLM",
                "prepare_inputs_labels_for_multimodal",
            )
        )
        self.assertIn("audit_mask_trace", arch_source)
        self.assertIn("record_consumed_mask", arch_source)
        self.assertIn("audit_provenance_trace", arch_source)
        self.assertIn("record_provenance_source", arch_source)
        self.assertIn("decoder_retained_visual_count", arch_source)
        self.assertNotIn("device='cpu'", arch_source)
        self.assertNotIn(".cpu()", arch_source)
        self.assertNotIn("torch.equal(previous, consumed)", arch_source)

        model_path = (
            REPO_ROOT
            / "LLaVA-NeXT"
            / "llava"
            / "model"
            / "language_model"
            / "llava_llama.py"
        )
        generate_source = ast.unparse(
            function_node(model_path, "LlavaLlamaForCausalLM", "generate")
        )
        self.assertIn("audit_mask_trace", generate_source)
        self.assertIn("audit_provenance_trace", generate_source)
        self.assertIn("audit_forward_stage", generate_source)


if __name__ == "__main__":
    unittest.main()

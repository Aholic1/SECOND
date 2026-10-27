"""Insert exactly one opt-in hook; retain a byte-exact backup; never replace server source wholesale."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "LLaVA-NeXT/llava/model/language_model/llava_recursion.py"
raw = path.read_bytes()
text = raw.decode("utf-8")
marker = "# STEP6_MASK_INTERVENTION"
if marker in text:
    print("Hook already installed; no changes")
else:
    needle = "                        applied_mask = applied_selection_mask(audit_condition, proposed_mask)"
    if text.count(needle) != 1:
        raise SystemExit("FAIL: source anchor differs; inspect server code before patching")
    backup = path.with_suffix(".py.pre_step6_backup")
    if backup.exists():
        raise SystemExit("FAIL: backup already exists; refusing overwrite")
    newline = "\r\n" if "\r\n" in text else "\n"
    addition = newline.join([
        needle,
        "                        # STEP6_MASK_INTERVENTION",
        "                        from lmms_eval.second_intervention import intervene_mask",
        "                        applied_mask = intervene_mask(applied_mask, audit_sample_id, token_idx, target_stage, audit_condition)",
    ])
    changed = text.replace(needle, addition)
    compile(changed, str(path), "exec")
    backup.write_bytes(raw)
    path.write_bytes(changed.encode("utf-8"))
    print("Installed opt-in hook. Backup:", backup)

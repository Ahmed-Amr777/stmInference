"""
CLAP-asm Normalizer for STM32F1 HAL Functions
===============================================
Takes extracted JSON (with addresses) and produces CLAP-ready normalized JSON.

Normalization rules:
  REMOVE:  commas, tabs, .n/.w suffixes, .word lines, nop, @ comments
  REPLACE: jump targets → INSTR<N>, call targets → EXTFUNC
  KEEP:    register names, constants, brackets, instruction names
"""

import re
import json
from pathlib import Path


def normalize_function(instructions: list[dict]) -> list[str]:
    """
    Normalize a list of {address, instruction} dicts to CLAP format.

    Input:  [{"address": 4, "instruction": "cbz\tr0, 28 <HAL_CRC_Init+0x24>"}, ...]
    Output: ["cbz r0 INSTR17", ...]
    """
    if not instructions:
        return []

    # ─── PASS 1: Filter out .word and nop, build address→instruction_number map ───

    real_instructions = []
    for item in instructions:
        addr = item["address"]
        text = item["instruction"]

        text = text.replace("\t", " ")

        if text.strip().startswith(".word"):
            continue

        if text.strip() == "nop":
            continue

        real_instructions.append((addr, text))

    # Build address → instruction number map (1-based like CLAP)
    addr_to_num = {}
    for i, (addr, _) in enumerate(real_instructions):
        addr_to_num[addr] = i + 1

    # ─── PASS 2: Normalize each instruction ───

    branch_re = re.compile(
        r'^(b|bl|bx|blx|beq|bne|bgt|bge|blt|ble|bhi|bhs|blo|bls|'
        r'bcc|bcs|bmi|bpl|bvs|bvc|cbz|cbnz)'
        r'(?:\.n|\.w)?'
        r'\s+(.*)',
        re.IGNORECASE
    )

    dotw_re = re.compile(r'^(\w+)\.(?:n|w)\b')

    normalized = []

    for addr, text in real_instructions:

        # Remove comments (@ ...)
        text = re.sub(r'\s*@.*$', '', text).strip()

        bm = branch_re.match(text)
        if bm:
            mnemonic = bm.group(1).lower()
            operands = bm.group(2).strip()

            # bx lr — return, keep as-is
            if mnemonic == "bx" and "lr" in operands:
                normalized.append("bx lr")
                continue

            # bl/blx — function call → EXTFUNC
            if mnemonic in ("bl", "blx"):
                normalized.append("bl EXTFUNC")
                continue

            # cbz/cbnz — register + target
            if mnemonic in ("cbz", "cbnz"):
                comma_pos = operands.find(",")
                if comma_pos >= 0:
                    reg_part = operands[:comma_pos].strip()
                    target_part = operands[comma_pos + 1:].strip()
                else:
                    parts = operands.split(None, 1)
                    reg_part = parts[0] if parts else operands
                    target_part = parts[1] if len(parts) > 1 else ""

                target_addr = _extract_target_addr(target_part)
                if target_addr is not None and target_addr in addr_to_num:
                    normalized.append(f"{mnemonic} {reg_part} INSTR{addr_to_num[target_addr]}")
                else:
                    normalized.append(f"{mnemonic} {_clean(operands)}")
                continue

            # Regular branches: b, beq, bne, bcc, etc.
            target_addr = _extract_target_addr(operands)
            if target_addr is not None and target_addr in addr_to_num:
                normalized.append(f"{mnemonic} INSTR{addr_to_num[target_addr]}")
            else:
                normalized.append(f"{mnemonic} EXTFUNC")
            continue

        # ── Non-branch instruction ──

        # Remove .w/.n suffix from and.w, orr.w, ldr.w, etc.
        dm = dotw_re.match(text)
        if dm:
            text = dm.group(1) + text[dm.end():]

        text = text.replace(",", "")
        text = re.sub(r'\s+', ' ', text).strip()

        normalized.append(text)

    return normalized


def _extract_target_addr(target_text: str) -> int | None:
    """Extract hex address from jump target string."""
    target_text = target_text.replace("short", "").strip()

    # "28 <HAL_CRC_Init+0x24>" or "0x54 <HAL_GPIO_Init+0x54>"
    m = re.match(r'(0x[0-9a-fA-F]+|[0-9a-fA-F]+)\s*<.*?>', target_text)
    if m:
        try:
            return int(m.group(1), 16)
        except ValueError:
            return None

    # Bare hex "28" or "1a" or "0x1c"
    m = re.match(r'^(0x[0-9a-fA-F]+|[0-9a-fA-F]+)$', target_text.strip())
    if m:
        try:
            return int(m.group(1), 16)
        except ValueError:
            return None

    return None


def _clean(operands: str) -> str:
    """Remove commas, annotations, extra whitespace."""
    operands = re.sub(r'<.*?>', '', operands)
    operands = operands.replace(",", "")
    return re.sub(r'\s+', ' ', operands).strip()


def normalize_batch(batch: list[list[dict]]) -> list[list[str]]:
    """
    Normalize a batch of functions in one call.

    Input:  [[{address, instruction}, ...], ...]   (one list per function)
    Output: [["cbz r0 INSTR3", ...], ...]          (one list per function)
    """
    return [normalize_function(instructions) for instructions in batch]


def normalize_data(data: dict) -> dict:
    """
    Normalize an already-parsed extraction dict to CLAP format.

    Input:  {"source_object": "...", "functions": [{"name": ..., "instructions": [...], ...}]}
    Output: same structure with instructions replaced by normalized strings.
    """
    clean_functions = []
    for fn in data["functions"]:
        norm = normalize_function(fn["instructions"])
        clean_functions.append({
            "name": fn["name"],
            "offset": fn["offset"],
            "size_bytes": fn["size_bytes"],
            "num_instructions": len(norm),
            "instructions": norm,
        })
    return {"source_object": data["source_object"], "functions": clean_functions}


def normalize_json_string(json_text: str) -> dict:
    """
    Normalize a raw JSON string (e.g. from an API response or in-memory buffer).

    Input:  JSON string with extraction format
    Output: normalized dict (call json.dumps() on it if you need a string back)
    """
    return normalize_data(json.loads(json_text))


def normalize_json(input_path: str, output_path: str = None) -> Path:
    """
    Normalize a JSON file on disk and write the result to another file.

    input_path:  path to extracted JSON file
    output_path: destination path (default: <stem>_clap.json next to input)
    Returns the output Path.
    """
    input_path = Path(input_path)
    clean_data = normalize_json_string(input_path.read_text())

    if output_path is None:
        output_path = input_path.parent / (input_path.stem + "_clap.json")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_data, indent=2))
    return output_path


if __name__ == "__main__":
    json_dir = Path("data/json")
    out_dir = json_dir / "normalized"
    for json_file in sorted(json_dir.glob("*.json")):
        out_path = out_dir / (json_file.stem + "_clap.json")
        result = normalize_json(str(json_file), str(out_path))
        print(f"Saved: {result}")

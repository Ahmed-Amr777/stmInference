from normaliztion import normalize_function, normalize_batch

# Small slice of HAL_GPIO_Init as it comes out of objdump (address + raw instruction).
# nop and .word are present to verify they get filtered out.
#
# addr  instr#  (after nop/.word filtered, 1-based)
# 0x00    1     stmdb
# 0x04    2     ldr
# 0x08    3     sub
# 0x0c    4     cmp
# 0x10    5     beq      → targets 0x54 = INSTR12
# 0x14    -     nop      filtered
# 0x18    -     .word    filtered
# 0x1c    6     movs
# 0x20    7     bl       → EXTFUNC
# 0x24    8     bx lr
# 0x28    9     cbz      → targets 0x54 = INSTR12
# 0x2c   10     and.w    .w stripped
# 0x30   11     ldr      @ comment stripped
# 0x54   12     bne      → targets 0x1c = INSTR6
HAL_GPIO_INIT = [
    {"address": 0x00, "instruction": "stmdb\tsp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}"},
    {"address": 0x04, "instruction": "ldr\tr4, [r1, #0]"},
    {"address": 0x08, "instruction": "sub\tsp, #20"},
    {"address": 0x0c, "instruction": "cmp\tr4, #0"},
    {"address": 0x10, "instruction": "beq.n\t0x54 <HAL_GPIO_Init+0x54>"},
    {"address": 0x14, "instruction": "nop"},
    {"address": 0x18, "instruction": ".word 0x40010800"},
    {"address": 0x1c, "instruction": "movs\tr2, #0"},
    {"address": 0x20, "instruction": "bl\t0x200 <HAL_RCC_GetHCLKFreq>"},
    {"address": 0x24, "instruction": "bx\tlr"},
    {"address": 0x28, "instruction": "cbz\tr0, 0x54 <HAL_GPIO_Init+0x54>"},
    {"address": 0x2c, "instruction": "and.w\tr6, r0, r4"},
    {"address": 0x30, "instruction": "ldr\tr4, [r1, #4]  @ load mode"},
    {"address": 0x54, "instruction": "bne\t0x1c <HAL_GPIO_Init+0x1c>"},
]

EXPECTED = [
    "stmdb sp! {r4 r5 r6 r7 r8 r9 sl fp lr}",
    "ldr r4 [r1 #0]",
    "sub sp #20",
    "cmp r4 #0",
    "beq INSTR12",
    "movs r2 #0",
    "bl EXTFUNC",
    "bx lr",
    "cbz r0 INSTR12",
    "and r6 r0 r4",
    "ldr r4 [r1 #4]",
    "bne INSTR6",
]


def test_normalize_function():
    result = normalize_function(HAL_GPIO_INIT)
    assert result == EXPECTED, (
        "\nExpected:\n" + "\n".join(EXPECTED) +
        "\n\nGot:\n" + "\n".join(result)
    )
    print("PASS  normalize_function — HAL_GPIO_Init snippet")


def test_normalize_batch():
    # Two functions in a batch — each has its own independent address space
    fn2 = [
        {"address": 0x00, "instruction": "push\t{lr}"},
        {"address": 0x02, "instruction": "bl\t0x100 <HAL_GPIO_WritePin>"},
        {"address": 0x06, "instruction": "pop\t{pc}"},
    ]
    results = normalize_batch([HAL_GPIO_INIT, fn2])
    assert len(results) == 2
    assert results[0] == EXPECTED
    assert results[1] == ["push {lr}", "bl EXTFUNC", "pop {pc}"]
    print("PASS  normalize_batch — 2 functions")


if __name__ == "__main__":
    test_normalize_function()
    test_normalize_batch()

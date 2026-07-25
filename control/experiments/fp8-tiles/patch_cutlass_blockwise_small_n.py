from pathlib import Path

path = Path("/opt/cutlass-src/include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl")
source = path.read_text()
replacements = {
    """  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_4,_2,_1>>, Layout<Shape<_2,_2,_1>>>;
""": """  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      cute::conditional_t<(size<1>(TileShape_MNK{}) >= 16), Layout<Shape<_4,_2,_1>>,
                          Layout<Shape<_8,_1,_1>>>,
      Layout<Shape<_2,_2,_1>>>;
""",
    """  using SmemCopyAtomB = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseF8f6f4>()), SmemAllocTypeB>;
""": """  using SmemCopyAtomB = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseF8f6f4,
                                                                                  size<1>(TileShape_MNK{})>()), SmemAllocTypeB>;
""",
}
for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one CUTLASS source match, found {count}: {old!r}")
    source = source.replace(old, new)
path.write_text(source)

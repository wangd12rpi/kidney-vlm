from __future__ import annotations

from kidney_vlm.genomics.extra_downloads import ExtraGenomicsDownloadSpec


EXTRA_GENOMICS_SPECS: list[ExtraGenomicsDownloadSpec] = [
    ExtraGenomicsDownloadSpec(
        key="dna_methylation_beta",
        subfolder="dna_methylation",
        data_category="DNA Methylation",
        data_types=["Methylation Beta Value"],
        experimental_strategies=["Methylation Array"],
        data_formats=["TXT"],
        workflow_types=["SeSAMe Methylation Beta Estimation"],
        access="open",
        description="Illumina 450K / EPIC level-3 beta value TSVs",
    ),
    ExtraGenomicsDownloadSpec(
        key="copy_number_gene",
        subfolder="copy_number_gene",
        data_category="Copy Number Variation",
        data_types=["Gene Level Copy Number"],
        experimental_strategies=["Genotyping Array", "WXS", "WGS"],
        data_formats=["TSV"],
        workflow_types=["ASCAT2", "ASCAT3", "GISTIC2_Gene_Level"],
        access="open",
        description="Gene-level copy number calls",
    ),
    ExtraGenomicsDownloadSpec(
        key="copy_number_segment",
        subfolder="copy_number_segment",
        data_category="Copy Number Variation",
        data_types=["Masked Copy Number Segment", "Allele-specific Copy Number Segment"],
        experimental_strategies=["Genotyping Array", "WXS"],
        data_formats=["TSV", "TXT"],
        workflow_types=["ASCAT2", "ASCAT3"],
        access="open",
        description="Masked segment-level copy number calls",
    ),
    ExtraGenomicsDownloadSpec(
        key="mutation_maf",
        subfolder="mutation_maf",
        data_category="Simple Nucleotide Variation",
        data_types=["Masked Somatic Mutation"],
        experimental_strategies=["WXS"],
        data_formats=["MAF"],
        workflow_types=[
            "Aliquot Ensemble Somatic Variant Merging and Masking",
            "MuSE Variant Aggregation and Masking",
            "Mutect2 Variant Aggregation and Masking",
            "VarScan2 Variant Aggregation and Masking",
            "Pindel Variant Aggregation and Masking",
            "SomaticSniper Variant Aggregation and Masking",
        ],
        access="open",
        description="Masked somatic mutation MAF files",
    ),
    ExtraGenomicsDownloadSpec(
        key="mirna_expression",
        subfolder="mirna_expression",
        data_category="Transcriptome Profiling",
        data_types=["miRNA Expression Quantification", "Isoform Expression Quantification"],
        experimental_strategies=["miRNA-Seq"],
        data_formats=["TSV", "TXT"],
        workflow_types=["BCGSC miRNA Profiling"],
        access="open",
        description="miRNA expression quantification files",
    ),
]


EXTRA_GENOMICS_SPEC_BY_KEY = {spec.key: spec for spec in EXTRA_GENOMICS_SPECS}

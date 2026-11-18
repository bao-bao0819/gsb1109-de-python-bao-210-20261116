# -*- coding: utf-8 -*-
"""Preview and execute a header-compatible merge of two VCF files.

Run with::

    python merge_vcf.py input_a.vcf input_b.vcf merged.vcf
"""

import sys

import vcfpy


def main():
    input_a, input_b, output = sys.argv[1:4]
    plan = vcfpy.merge_vcf_files([input_a, input_b], output, source_names=["A", "B"])

    print("compatible:", "yes" if plan.compatible else "no")
    for rename in plan.renames:
        print(
            "rename {scope} {old} (source {source}) -> {new}: {reason}".format(
                scope=rename.scope,
                old=rename.old_id,
                source=rename.source_name,
                new=rename.new_id,
                reason=rename.reason,
            )
        )
    print("wrote:", output)


if __name__ == "__main__":
    main()

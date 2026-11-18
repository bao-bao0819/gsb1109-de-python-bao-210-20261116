import io
import json
import os
import tempfile
import unittest

from gspread.validation import (
    A1Range,
    CellRule,
    CellStatus,
    OptionCategory,
    OptionSource,
    RuleMode,
    RuleType,
    ValidationAudit,
    ValidationBatch,
    audit_from_dict,
    load_batch,
    main,
)


class ValidationRangeTest(unittest.TestCase):
    def test_relative_range_is_rebased_like_a_spreadsheet(self):
        rule = CellRule("D2:D4", list_range="B2:B2")
        resolved = [
            str(rule.for_cell(row, 4).resolve_list_range(row, 4))
            for row in range(2, 5)
        ]
        self.assertEqual(resolved, ["B2", "B3", "B4"])

    def test_mixed_absolute_range(self):
        rule = CellRule("D2:D4", list_range="$B2:$B2")
        resolved = [
            str(rule.for_cell(row, 4).resolve_list_range(row, 4))
            for row in range(2, 5)
        ]
        self.assertEqual(resolved, ["$B2", "$B3", "$B4"])

    def test_absolute_range_does_not_move(self):
        rule = CellRule("D2:D4", list_range="$B$2:$B$2")
        resolved = [
            str(rule.for_cell(row, 4).resolve_list_range(row, 4))
            for row in range(2, 5)
        ]
        self.assertEqual(resolved, ["$B$2", "$B$2", "$B$2"])

    def test_relative_range_keeps_column_offset_per_cell(self):
        rule = CellRule("D2:F2", list_range="A2:A2")
        self.assertEqual(
            str(rule.for_cell(2, 6).resolve_list_range(2, 6)), "C2"
        )

    def test_origin_can_be_declared(self):
        rule = CellRule("D5", list_range="B1:B1", origin="D2")
        self.assertEqual(str(rule.for_cell(5, 4).resolve_list_range(5, 4)), "B4")

    def test_parse_and_format_sheet_names(self):
        parsed = A1Range.parse("'My Sheet'!$B$2:C4")
        self.assertEqual(parsed.sheet, "My Sheet")
        self.assertEqual(str(parsed), "'My Sheet'!$B$2:C4")

    def test_unbounded_range_rejected_for_rules(self):
        with self.assertRaises(ValueError):
            CellRule("D:D", list_range="B:B")


class ValidationModeTest(unittest.TestCase):
    def _audit(self, mode):
        rules = [CellRule("D2", mode=mode, values=["A", "B"])]
        audit = ValidationAudit(ValidationBatch(rules))
        return audit.validate({"D2": "C"})[0]

    def test_reject_mode_marks_invalid(self):
        result = self._audit(RuleMode.reject)
        self.assertEqual(result.status, CellStatus.invalid)
        self.assertTrue(result.rejected)
        self.assertEqual(result.violating_values, ["C"])

    def test_warn_mode_only_warns(self):
        result = self._audit(RuleMode.warn)
        self.assertEqual(result.status, CellStatus.warned)
        self.assertTrue(result.warned)
        self.assertFalse(result.rejected)

    def test_valid_value_passes_in_both_modes(self):
        for mode in (RuleMode.reject, RuleMode.warn):
            rules = [CellRule("D2", mode=mode, values=["A", "B"])]
            result = ValidationAudit(ValidationBatch(rules)).validate(
                {"D2": "A"}
            )[0]
            self.assertEqual(result.status, CellStatus.valid)


class ValidationSourceTest(unittest.TestCase):
    def test_category_specific_ranges_are_used(self):
        source = OptionSource(
            "colors",
            categories={
                "gold": {"values": ["Yellow", "Orange"], "range": "A2:A3"},
                "silver": {"values": ["Grey", "White"], "range": "B2:B3"},
            },
        )
        rules = [
            CellRule("D2", source="colors", category="gold"),
            CellRule("D3", source="colors", category="silver"),
        ]
        results = ValidationAudit(ValidationBatch(rules, [source])).validate(
            {"D2": "Yellow", "D3": "Grey"}
        )
        self.assertEqual([r.status for r in results], [CellStatus.valid] * 2)
        self.assertEqual(results[0].resolved_range, A1Range.parse("A2:A3"))
        self.assertEqual(results[1].resolved_range, A1Range.parse("B2:B3"))
        self.assertEqual(results[0].allowed, ["Yellow", "Orange"])
        self.assertEqual(results[1].allowed, ["Grey", "White"])

    def test_incomplete_source_yields_pending_not_invalid(self):
        source = OptionSource("colors", values=["Red", "Blue"], complete=False)
        rule = CellRule("D2", source="colors")
        result = ValidationAudit(ValidationBatch([rule], [source])).validate(
            {"D2": "Green"}
        )[0]
        self.assertEqual(result.status, CellStatus.pending)
        self.assertEqual(result.pending_reason, "incomplete-source")
        self.assertFalse(result.rejected)
        self.assertEqual(result.violating_values, [])

    def test_incomplete_source_is_pending_even_for_known_values(self):
        # The locked read may have missed values, so no conclusion is safe.
        source = OptionSource("colors", values=["Red", "Blue"], complete=False)
        rule = CellRule("D2", source="colors")
        result = ValidationAudit(ValidationBatch([rule], [source])).validate(
            {"D2": "Red"}
        )[0]
        self.assertEqual(result.status, CellStatus.pending)
        self.assertEqual(result.pending_reason, "incomplete-source")

    def test_incomplete_category_yields_pending(self):
        source = OptionSource(
            "colors",
            complete=True,
            categories={
                "gold": {"values": ["Yellow"], "complete": True},
                "silver": {"values": ["Grey"], "complete": False},
            },
        )
        rules = [
            CellRule("D2", source="colors", category="gold"),
            CellRule("D3", source="colors", category="silver"),
        ]
        results = ValidationAudit(ValidationBatch(rules, [source])).validate(
            {"D2": "Purple", "D3": "Purple"}
        )
        self.assertEqual(results[0].status, CellStatus.invalid)
        self.assertEqual(results[1].status, CellStatus.pending)

    def test_value_not_locked_source_is_pending(self):
        rule = CellRule("D2", source="unknown")
        result = ValidationAudit(ValidationBatch([rule])).validate({"D2": "A"})[0]
        self.assertEqual(result.status, CellStatus.pending)
        self.assertEqual(result.pending_reason, "source-missing")


class ValidationIncrementalTest(unittest.TestCase):
    def _batch(self):
        source = OptionSource(
            "products",
            complete=True,
            categories={
                "food": {
                    "values": ["Apple", "Bread"],
                    "range": "A2:A3",
                    "revision": 1,
                },
                "tools": {
                    "values": ["Hammer", "Saw"],
                    "range": "B2:B3",
                    "revision": 1,
                },
            },
        )
        rules = [
            CellRule("D2", source="products", category="food"),
            CellRule("D3", source="products", category="food"),
            CellRule("D4", source="products", category="tools"),
        ]
        return ValidationBatch(rules, [source])

    def test_only_dependent_cells_are_revalidated(self):
        batch = self._batch()
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Apple", "D3": "Bread", "D4": "Hammer"})
        before = {result.cell: id(result) for result in audit.results}

        products = batch.source("products")
        products.replace_category(
            OptionCategory(
                "food",
                values=["Apple", "Bread", "Cake"],
                range=A1Range.parse("A2:A4"),
                revision=2,
            )
        )
        revalidated = audit.revalidate_changed({"D3": "Cake"})
        self.assertEqual([r.cell for r in revalidated], ["D2", "D3"])
        self.assertEqual(revalidated[1].status, CellStatus.valid)
        after = {result.cell: id(result) for result in audit.results}
        self.assertEqual(before["D4"], after["D4"])
        self.assertNotEqual(before["D2"], after["D2"])

    def test_unchanged_category_does_not_revalidate(self):
        batch = self._batch()
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Apple", "D3": "Bread", "D4": "Hammer"})
        audit.revalidate_changed()
        self.assertEqual(audit.revalidate_changed(), [])

    def test_force_category_revalidation(self):
        batch = self._batch()
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Apple", "D3": "Bread", "D4": "Hammer"})
        results = audit.revalidate_category("products", "tools", force=True)
        self.assertEqual([result.cell for result in results], ["D4"])

    def test_dependents_are_scoped_by_category(self):
        batch = self._batch()
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Apple", "D3": "Bread", "D4": "Hammer"})
        self.assertEqual(audit.dependents("products", "food"), ["D2", "D3"])
        self.assertEqual(audit.dependents("products", "tools"), ["D4"])
        self.assertEqual(audit.dependents("products"), ["D2", "D3", "D4"])

    def test_range_locked_cells_are_revalidated(self):
        source = OptionSource("colors", categories={
            "gold": {"values": ["Yellow"], "range": "A2:A3", "revision": 1}
        })
        rule = CellRule("D2", list_range="A2:A3")
        batch = ValidationBatch([rule], [source])
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Purple"})
        self.assertEqual(audit.dependents("colors", "gold"), ["D2"])

    def test_replacing_source_revalidates_only_changed_category(self):
        batch = self._batch()
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Apple", "D3": "Bread", "D4": "Hammer"})
        batch.replace_source(
            OptionSource(
                "products",
                categories={
                    "food": {
                        "values": ["Apple", "Bread", "Cake"],
                        "range": "A2:A4",
                    },
                    "tools": {
                        "values": ["Hammer", "Saw"],
                        "range": "B2:B3",
                    },
                },
            )
        )
        revalidated = audit.revalidate_changed()
        self.assertEqual([r.cell for r in revalidated], ["D2", "D3"])

    def test_source_replacement_without_changes_is_noop(self):
        batch = self._batch()
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Apple", "D3": "Bread", "D4": "Hammer"})
        batch.replace_source(
            OptionSource(
                "products",
                categories={
                    "food": {"values": ["Apple", "Bread"], "range": "A2:A3"},
                    "tools": {"values": ["Hammer", "Saw"], "range": "B2:B3"},
                },
            )
        )
        self.assertEqual(audit.revalidate_changed(), [])

    def test_range_that_becomes_locked_is_revalidated(self):
        rule = CellRule("D2", list_range="A2:A3")
        batch = ValidationBatch([rule])
        audit = ValidationAudit(batch)
        audit.validate({"D2": "Purple"})
        self.assertEqual(audit.results[0].status, CellStatus.pending)
        batch.replace_source(
            OptionSource(
                "colors",
                categories={
                    "gold": {"values": ["Yellow"], "range": "A2:A3"}
                },
            )
        )
        revalidated = audit.revalidate_changed()
        self.assertEqual([r.cell for r in revalidated], ["D2"])
        self.assertEqual(audit.revalidate_changed(), [])
        self.assertEqual(audit.dependents("colors", "gold"), ["D2"])


class ValidationRuleTest(unittest.TestCase):
    def _run(self, rule, value):
        return ValidationAudit(ValidationBatch([rule])).validate({"D2": value})[0]

    def test_number_bounds(self):
        rule = CellRule(
            "D2", rule_type=RuleType.number, min_value=10, max_value=20
        )
        self.assertEqual(self._run(rule, 15).status, CellStatus.valid)
        self.assertEqual(self._run(rule, 5).status, CellStatus.invalid)
        self.assertEqual(self._run(rule, 25).status, CellStatus.invalid)
        self.assertEqual(self._run(rule, "abc").status, CellStatus.invalid)

    def test_text_length_and_allowed_set(self):
        rule = CellRule(
            "D2",
            rule_type=RuleType.text,
            min_value=2,
            max_value=4,
            values=["ab", "abcd"],
        )
        self.assertEqual(self._run(rule, "abc").status, CellStatus.invalid)
        self.assertEqual(self._run(rule, "ab").status, CellStatus.valid)

    def test_date_bounds(self):
        rule = CellRule(
            "D2",
            rule_type=RuleType.date,
            min_value="2026-01-01",
            max_value="2026-12-31",
        )
        self.assertEqual(self._run(rule, "2026-11-19").status, CellStatus.valid)
        self.assertEqual(self._run(rule, "2027-01-01").status, CellStatus.invalid)

    def test_blank_policy(self):
        required = CellRule("D2", values=["A"], allow_blank=False)
        optional = CellRule("D2", values=["A"], allow_blank=True)
        self.assertEqual(self._run(required, "").status, CellStatus.invalid)
        self.assertEqual(self._run(optional, "").status, CellStatus.empty)

    def test_custom_formula_is_pending(self):
        rule = CellRule("D2", rule_type=RuleType.custom_formula, options="=A1>0")
        result = self._run(rule, "x")
        self.assertEqual(result.status, CellStatus.pending)
        self.assertEqual(result.pending_reason, "formula-pending")


class ValidationImportExportTest(unittest.TestCase):
    def test_report_contains_constraints_and_violations(self):
        source = OptionSource("colors", values=["Red"])
        rule = CellRule("D2", source="colors", mode=RuleMode.warn)
        audit = ValidationAudit(ValidationBatch([rule], [source]))
        audit.validate({"D2": "Blue"})
        report = audit.to_report()
        self.assertEqual(report["cells"][0]["status"], "warned")
        self.assertEqual(report["cells"][0]["violating_values"], ["Blue"])
        self.assertEqual(
            report["constraints"]["D2"]["options"], ["Red"]
        )
        self.assertEqual(report["violations"]["D2"], ["Blue"])

    def test_json_round_trip(self):
        data = {
            "name": "batch",
            "sources": [
                {
                    "name": "colors",
                    "complete": False,
                    "categories": {
                        "gold": {"values": ["Yellow"], "range": "A2:A3"}
                    },
                }
            ],
            "rules": [
                {
                    "cell": "D2",
                    "source": "colors",
                    "category": "gold",
                    "mode": "warn",
                }
            ],
            "values": {"D2": "Yellow"},
        }
        audit = audit_from_dict(data)
        self.assertEqual(audit.results[0].status, CellStatus.pending)

        data["values"] = {"D2": "Grey"}
        audit = audit_from_dict(data)
        self.assertEqual(audit.results[0].status, CellStatus.pending)

        data["sources"][0]["complete"] = True
        audit = audit_from_dict(data)
        self.assertEqual(audit.results[0].status, CellStatus.warned)
        batch = ValidationBatch.from_dict(data)
        restored = load_batch(io.StringIO(json.dumps(data)))
        self.assertEqual(batch.to_dict()["rules"], restored.to_dict()["rules"])

    def test_csv_export(self):
        rule = CellRule("D2", values=["A"])
        audit = ValidationAudit(ValidationBatch([rule]))
        audit.validate({"D2": "B"})
        csv_text = audit.to_csv()
        self.assertIn("cell,status,mode", csv_text)
        self.assertIn("D2,invalid,reject", csv_text)

    def test_cli_writes_report(self):
        data = {
            "sources": [{"name": "colors", "values": ["Red"]}],
            "rules": [{"cell": "D2", "source": "colors"}],
            "values": {"D2": "Blue"},
        }
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "batch.json")
            output_path = os.path.join(directory, "report.json")
            with open(source_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            self.assertEqual(
                main([source_path, "-o", output_path]), 0
            )
            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(report["cells"][0]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()

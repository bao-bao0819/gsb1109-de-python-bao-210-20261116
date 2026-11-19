#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
import os
import struct
import unittest

from Registry import Registry
from Registry import RegistryLog
from Registry import Incremental


SAMPLE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "testing", "reg_samples")
_CACHE = {}


def _load(name, log_order):
    cache_key = (name, log_order)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    base = os.path.join(SAMPLE_ROOT, name)
    with open(os.path.join(base, "SYSTEM"), "rb") as handle:
        raw = handle.read()

    primary = io.BytesIO(raw)
    logs = {}
    for log_name in ("LOG1", "LOG2"):
        primary.seek(0)
        logs[log_name] = RegistryLog.RegistryLog(
            primary, os.path.join(base, "SYSTEM." + log_name))

    primary.seek(0)
    sequence = logs[log_order[0]].recover_hive()
    primary.seek(0)
    logs[log_order[1]].recover_hive_continue(sequence + 1)
    recovered = primary.getvalue()

    regions = []
    for log_name in log_order:
        log = RegistryLog.RegistryLog(io.BytesIO(raw),
                                      os.path.join(base, "SYSTEM." + log_name))
        regions.extend(Incremental.dirty_pages_from_log(log))

    result = (raw, recovered, regions)
    _CACHE[cache_key] = result
    return result


def _plan(name, log_order):
    cache_key = ("plan", name, log_order)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    raw, recovered, regions = _load(name, log_order)
    index = Incremental.build_index(Registry.Registry(io.BytesIO(raw)))
    baseline = Incremental.collect_config(Registry.Registry(io.BytesIO(raw)))
    plan = Incremental.plan_recollection(index, regions)
    replayed = Incremental.replay_recollection(
        baseline, plan, Registry.Registry(io.BytesIO(recovered)))
    full = Incremental.collect_config(Registry.Registry(io.BytesIO(recovered)))
    result = (index, baseline, plan, replayed, full)
    _CACHE[cache_key] = result
    return result


class TestDependencyIndex(unittest.TestCase):
    def setUp(self):
        self.raw = open(os.path.join(SAMPLE_ROOT, "new_log_1", "SYSTEM"),
                        "rb").read()
        self.registry = Registry.Registry(io.BytesIO(self.raw))

    def test_value_data_fragments(self):
        key = self.registry.root().find_key(
            r"ControlSet001\Control\ProductOptions")
        value = key.value("ProductPolicy")._vkrecord
        fragments = value.data_fragments()
        self.assertTrue(value.is_large_data())
        self.assertGreater(len(fragments), 2)
        self.assertEqual(value.data_length(), 23712)
        # The first two fragments are the DB record and indirect block.
        self.assertEqual(b"db", self.raw[fragments[0][0] + 0x4:
                                         fragments[0][0] + 0x6])

    def test_value_data_location_is_indexed(self):
        index = Incremental.build_index(self.registry)
        dependency = index.value_dependency(
            self.registry.root().path() +
            r"\ControlSet001\Control\ProductOptions",
            "ProductPolicy")
        self.assertIsNotNone(dependency)
        self.assertEqual(4, len(dependency.fragments))
        for start, size in dependency.fragments:
            self.assertTrue(any(page in index._by_page
                                for page in range(
                                    start - start % 0x1000,
                                    start + size, 0x1000)))

    def test_empty_dirty_pages_give_empty_plan(self):
        index = Incremental.build_index(self.registry)
        plan = Incremental.plan_recollection(index, [])
        self.assertTrue(plan.is_empty)
        self.assertEqual([], plan.items)


class TestIncrementalReplay(unittest.TestCase):
    def test_new_log_1_replay_matches_full_collection(self):
        index, baseline, plan, replayed, full = _plan(
            "new_log_1", ("LOG1", "LOG2"))
        self.assertTrue(plan.items)
        self.assertTrue(plan.dirty_pages)
        self.assertEqual(full, replayed)

    def test_new_log_2_replay_uses_log_order(self):
        index, baseline, plan, replayed, full = _plan(
            "new_log_2", ("LOG2", "LOG1"))
        self.assertTrue(plan.items)
        self.assertEqual(full, replayed)

    def test_every_recollect_item_has_a_trigger(self):
        index, baseline, plan, replayed, full = _plan(
            "new_log_1", ("LOG1", "LOG2"))
        for item in plan.items:
            self.assertTrue(item.triggers, item)
            for trigger in item.triggers:
                self.assertTrue(trigger.kind, item)
                self.assertIsNotNone(trigger.page, item)

    def test_large_value_fragment_change_is_localized(self):
        index, baseline, plan, replayed, full = _plan(
            "new_log_1", ("LOG1", "LOG2"))
        found = False
        for item in plan.items:
            if item.value_name == "ProductPolicy":
                found = True
                fields = [field for trigger in item.triggers
                          for field in trigger.fields]
                self.assertTrue(any(field.startswith("data_fragment_")
                                    for field in fields), fields)
        self.assertTrue(found)

    def test_pending_items_are_not_silently_resolved(self):
        index, baseline, plan, replayed, full = _plan(
            "new_log_1", ("LOG1", "LOG2"))
        self.assertTrue(plan.pending)
        for item in plan.pending:
            self.assertEqual(Incremental.STATUS_PENDING, item.status)
            self.assertTrue(item.triggers)

    def test_export_matches_full_export(self):
        raw, recovered, regions = _load("new_log_1", ("LOG1", "LOG2"))
        collector = Incremental.IncrementalCollector(
            Registry.Registry(io.BytesIO(raw)))
        plan = collector.plan(regions)
        collector.replay(plan, Registry.Registry(io.BytesIO(recovered)))
        self.assertEqual(
            Incremental.collect_config(
                Registry.Registry(io.BytesIO(recovered))).export(),
            collector.export())
        self.assertTrue(collector.verify(
            Registry.Registry(io.BytesIO(recovered))))


class TestIdentityInvalidation(unittest.TestCase):
    def test_subkey_pointer_to_unknown_cell_is_pending(self):
        raw = open(os.path.join(SAMPLE_ROOT, "new_log_1", "SYSTEM"),
                   "rb").read()
        index = Incremental.build_index(Registry.Registry(io.BytesIO(raw)))
        root = index.key_dependency(Registry.Registry(
            io.BytesIO(raw)).root().path())
        # Simulate a directory change that points at an unindexed NK cell.
        patched = bytearray(raw)
        list_offset = 0x1000 + struct.unpack_from(
            "<I", raw, root.record_offset + 0x1C)[0]
        page = list_offset - (list_offset % 0x1000)
        patched[list_offset + 0x8:list_offset + 0xC] = struct.pack(
            "<I", 0xB0000)
        region = Incremental.DirtyPageRegion(
            page, 0x1000, data=bytes(patched[page:page + 0x1000]))
        plan = Incremental.plan_recollection(index, [region])
        pending = [item for item in plan.items
                   if item.status == Incremental.STATUS_PENDING]
        self.assertTrue(any("requires rediscovery" in (item.detail or "")
                            for item in pending), plan.explain())

    def test_value_pointer_to_unknown_cell_is_pending(self):
        raw = open(os.path.join(SAMPLE_ROOT, "new_log_1", "SYSTEM"),
                   "rb").read()
        index = Incremental.build_index(Registry.Registry(io.BytesIO(raw)))
        key = Registry.Registry(io.BytesIO(raw)).root().find_key(
            r"ControlSet001\Control")
        list_offset = 0x1000 + key._nkrecord.unpack_dword(0x28)
        page = list_offset - (list_offset % 0x1000)
        patched = bytearray(raw)
        patched[list_offset + 0x4:list_offset + 0x8] = struct.pack(
            "<I", 0x300000)
        region = Incremental.DirtyPageRegion(
            page, 0x1000, data=bytes(patched[page:page + 0x1000]))
        plan = Incremental.plan_recollection(index, [region])
        pending = [item for item in plan.items
                   if item.status == Incremental.STATUS_PENDING]
        self.assertTrue(any("new value pointer" in (item.detail or "")
                            for item in pending), plan.explain())

    def test_missing_payload_never_invents_removals(self):
        raw = open(os.path.join(SAMPLE_ROOT, "new_log_1", "SYSTEM"),
                   "rb").read()
        index = Incremental.build_index(Registry.Registry(io.BytesIO(raw)))
        key = Registry.Registry(io.BytesIO(raw)).root().find_key(
            r"ControlSet001\Control")
        page = (0x1000 + key._nkrecord.unpack_dword(0x28)) & ~0xFFF
        plan = Incremental.plan_recollection(index, [(page, 0x1000)])
        self.assertEqual([], plan.removals)
        self.assertTrue(plan.pending)

    def test_renamed_value_loses_old_identity(self):
        raw = open(os.path.join(SAMPLE_ROOT, "new_log_1", "SYSTEM"),
                   "rb").read()
        registry = Registry.Registry(io.BytesIO(raw))
        index = Incremental.build_index(registry)
        dependency = index.value_dependency(
            registry.root().path() + r"\ControlSet001\Control", "CurrentUser")
        page = dependency.start - (dependency.start % 0x1000)
        patched = bytearray(raw)
        replacement = b"Z" * dependency.name_length
        start = dependency.record_offset + 0x14
        patched[start:start + len(replacement)] = replacement
        plan = Incremental.plan_recollection(
            index,
            [Incremental.DirtyPageRegion(
                page, 0x1000, data=bytes(patched[page:page + 0x1000]))])
        self.assertFalse(any(item.value_name == "CurrentUser" and item.status ==
                             Incremental.STATUS_RECOLLECT
                             for item in plan.items))
        self.assertTrue(any(item.pending and item.value_name == "CurrentUser"
                            for item in plan.items), plan.explain())
        self.assertTrue(any(item.pending and item.value_name == replacement.decode("ascii")
                            for item in plan.items), plan.explain())
        self.assertTrue(any(removal.value_name == "CurrentUser"
                            for removal in plan.removals))

    def test_value_type_change_is_content_not_identity(self):
        raw = open(os.path.join(SAMPLE_ROOT, "new_log_1", "SYSTEM"),
                   "rb").read()
        registry = Registry.Registry(io.BytesIO(raw))
        index = Incremental.build_index(registry)
        dependency = index.value_dependency(
            registry.root().path() + r"\ControlSet001\Control", "CurrentUser")
        page = dependency.start - (dependency.start % 0x1000)
        patched = bytearray(raw)
        patched[dependency.record_offset + 0xC:
                dependency.record_offset + 0x10] = struct.pack("<I", 3)
        plan = Incremental.plan_recollection(
            index,
            [Incremental.DirtyPageRegion(
                page, 0x1000, data=bytes(patched[page:page + 0x1000]))])
        items = [item for item in plan.items
                 if item.value_name == "CurrentUser"]
        self.assertEqual(1, len(items), plan.explain())
        self.assertFalse(items[0].pending)


if __name__ == "__main__":
    unittest.main(verbosity=2)

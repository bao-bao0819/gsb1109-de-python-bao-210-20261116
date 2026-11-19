# 清洗候选的完整约束影响分析

删除一条关系（三元组）不是只影响"它触发的那条违规"。同一路径上的
`sh:minCount / sh:maxCount / sh:datatype / sh:minInclusive …` 共享同一组值，
一次删除会同时改变计数类与取值类约束的判定；多个候选组合时还会产生
单候选分析里不存在的新违规。本目录用 pySHACL 对每个候选做**真实重验证**，
输出带证据、带确定性指纹的影响报告，而不是静态统计消失项。

## 方法（与需求逐条对应）

1. **基线验证**：对原始 `data.ttl` 跑一次 SHACL，提取每条违规的
   `focusNode / resultPath / sourceConstraintComponent / sourceShape / value / resultMessage`，
   并计算不含消息文案的确定性指纹（消息改写不改变违规身份）。
2. **候选即真实删除 + 真实重验证**：每个候选在全新数据图副本上执行
   `Graph.remove`，删除不存在的三元组会直接报错，然后重新调用 `pyshacl.validate`。
   差异（消失 / 新增 / 保留）全部由两次验证的指纹集合相减得到。
3. **完整相关约束，不只看消失项**：对删除三元组所属的「焦点节点 + 路径」，
   列出该 PropertyShape 上挂载的全部约束及其参数，逐条给出 `FAIL/PASS -> FAIL/PASS`。
   通过的约束（如 `minCount`）也在视图内，共同依赖由此显式可见。
4. **组合不可相加**：组合候选整体删除、整体重验证得到「实测」结果；
   再把各单候选的消失/新增指纹做并集得到「朴素求和」，输出四类差异证据。
5. **可重建同一预期差异**：`expected_diff.json` 是基准报告，
   `--check` 用规范化 JSON 全量比对，任何漂移以非零退出码失败。

## 场景与证据（`scenario/`）

形状（`shapes.ttl`）在 `:age` 路径上挂了四个共享约束：
`MinCount(1)`、`MaxCount(1)`、`Datatype(xsd:integer)`、`MinInclusive(18)`。

数据（`data.ttl`）：alice 有 `:age 19`（合法）与 `:age "17"`（字符串且低于 18）；
bob 缺 `:name`；carol 完全合规（对照组）。

基线共 4 条违规，候选实测结果：

| 候选 | 删除 | 消失 | 新增 | 保留 | 关键约束前后状态（焦点:路径） |
|------|------|------|------|------|-------------------------------|
| C1 | alice `:age "17"` | 3 | 0 | 1 | MaxCount/Datatype/MinInclusive 全 `FAIL→PASS`，MinCount 保持 `PASS→PASS` |
| C2 | alice `:age 19` | 1 | 0 | 3 | 仅 MaxCount `FAIL→PASS`；Datatype/MinInclusive 仍 FAIL |
| C3 | bob `:age 20` | 0 | 1 | 4 | MinCount `PASS→FAIL`（必填缺失，净违规 +1） |
| COMBO C1+C2 | alice 两条 `:age` | 3 | **1** | — | 朴素求和预测新增 0 条，实测新增 alice 的 MinCount 违规 |

要点：

- C1 就是「删一条消三违」的候选，但报告同时证明它没有破坏 `minCount`，
  且 bob 的既有违规原样保留——结论建立在完整约束视图上。
- C3 演示「也可能造成另一个必填缺失」：消失项为 0，必须重验证才能发现新增。
- COMBO 演示不可加性：C1、C2 单独执行都不引入违规，朴素求和预测净效果 −4，
  但两条一起删导致 alice 无 `:age`，实测新增一条 `MinCount` 违规
  （证据在报告的 `nonAdditivity.introducedActualButMissedByNaiveSum`）。
- carol 从不出现在任何结果中，证明分析不会波及无关焦点节点。

## 运行方式

```bash
# 生成 report.json 并打印文本摘要
.venv/bin/python cleaning_analysis/analyzer.py

# 重建并校验同一份预期验证差异（CI/执行前门禁）
.venv/bin/python cleaning_analysis/analyzer.py --check cleaning_analysis/expected_diff.json

# 回归测试
.venv/bin/pytest cleaning_analysis/test_analyzer.py -q
```

## 接到真实数据

替换 `scenario/` 下三个文件即可，分析器无需改动：

- `shapes.ttl`：真实 SHACL 形状；
- `data.ttl`：待清洗数据图；
- `candidates.json`：候选删除的三元组（CURIE 前缀在 `prefixes` 声明，
  字面量可用 `datatype` 指定数据类型，URI 对象用 `"type": "uri"`）。

重新生成报告、人工复核证据后，再覆盖 `expected_diff.json` 作为新的基准。

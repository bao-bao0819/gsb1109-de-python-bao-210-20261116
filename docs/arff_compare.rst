ARFF 分层类别占比对比
====================

`arff_compare` 用两期 ARFF 文件、给定的分层属性、类别等价映射和观测权重，
分别给出：

- 各期实际加权类别占比；
- 统一到基准期层权重后的占比（直接标准化），以及各层贡献；
- “只改了层权重”这一解释的可达范围，以及一个能复现目标占比的具体补全。

它不会把缺失层、未知类别或缺失类别当作 0。所有点估计都对应一个写明名称的
补全；范围（bounds）不依赖任何补全。

快速开始
--------

::

    python -m arff_compare 基准期.arff 对比期.arff \
        --strata region,channel \
        --class-attribute outcome \
        --weight-attribute survey_weight \
        --mapping classes.json \
        --format text

``classes.json`` 支持三种写法::

    {"positive": ["yes", "Y"], "negative": ["no", "N"]}

    {"classes": ["positive", "negative"],
     "map": {"yes": "positive", "no": "negative"}}

    [{"group": "positive", "values": ["yes", "Y"]}]

省略 ``--mapping`` 时使用两期 ARFF 头中声明标签的恒等映射。

输入约定
--------

- **分层**：``--stratum`` 可重复，或用 ``--strata a,b`` 表示复合层。层键由这些
  属性的值组成。
- **类别**：类别属性应为 nominal；若确为 ``STRING``，必须显式给出
  ``--mapping`` 并加 ``--allow-string-class``。
- **权重**：``--weight-attribute`` 指定观测权重；省略时每行权重为 1。
  权重缺失默认报错；``--missing-weight one`` 或 ``mean`` 是显式规则，会写进报告。
  负权重与非有限权重直接报错。
- **缺失层**：分层属性为 ``?`` 的行不会被塞进某个层，而是单独记为
  ``(missing stratum)``。
- **未知类别**：不在映射内的标签、以及缺失类别，按层记为 residual mass，并在
  ``missing`` 中按标签列出。它们不会被丢弃或并入第一个类别。

读数方式
--------

``actual``
    各期实际加权占比。``completed_shares`` 是按层内已知映射质量比例分摊
    residual 后的点估计；``bounds`` 是每类的边际可达范围。
    ``known_mapped_shares`` 是不分摊 residual 的已映射质量。

``standardized.primary``
    对比期按基准期各层总权重重新加权。只使用两期都出现的层；覆盖率同时给出。
    同一层对比期的类别率，其未知类别质量按 ``[0, residual]`` 计入。
    ``primary.point`` 只在共同层上可识别；没有共同层时为 ``null``。

``standardized.full_baseline``
    分母改为全部基准期权重。单边层按未知类别率 ``[0, 1]`` 计入，因此范围通常
    更宽。

``standardized.strata``
    逐层列出基准权重份额、对比期类别率和该层对标准化占比的贡献及范围。

``stratumwise_equality``
    逐层检验“层内类别率不变”是否可行：每层每类取两期区间的交集。
    全部共同层都可行时，才存在一个补全使两期层内类别率完全一致。
    只在单边出现的层无法检验，单独列出，不强行视为一致。

``fixed_rates``
    “只改层权重”的反事实合计：对比期权重 × 基准期层内类别率。
    仅在对比期出现的层保留其观测权重，基准期类别率取 ``[0, 1]``。
    这是汇总层面的必要检查，但不能替代 ``stratumwise_equality``：
    各层方向相反的变化可能在合计上互相抵消。

``decomposition``
    两期都出现的层给出 Kitagawa 分解：``mix = (wB-wA)*pA``、
    ``rate = wA*(pB-pA)``、``interaction = (wB-wA)*(pB-pA)``。
    只在单边出现的层不参与这些合计，也不被当作 0 贡献，而是列入
    ``noncomparable_change_point`` 或在表中标为单边。

``explanation``
    每类给出实际变化、标准化变化、与固定率反事实的差，以及：

    - ``weight_only_change_not_ruled_out``：每个共同层都存在两期一致的类别率，
      并附具体补全；
    - ``weight_only_change_ruled_out``：至少一个共同层无法取到一致类别率；
      ``strata_blocking_identity`` 列出这些层，``stratum_rate_gaps`` 给出
      每层最小缺口；
    - ``not_identifiable``：没有共同层，或区间本身覆盖整个 ``[0, 1]``。

    “未被排除”只表示数据不足以否定该解释，不表示它是唯一解释。

``missing``
    各期缺失层权重、缺失类别权重、未知标签权重，以及单边层清单。

关于范围
--------

范围都是每类的边际范围。未知质量一次只能分给一个类别，因此不同类别的上界
不能同时达到。报告不在类别之间强行归一化；需要联合可行的结论时，应以
``completed_shares`` 的具体补全为准，或把 ``bounds`` 当作保守的逐类范围。

示例
----

``examples/arff_compare`` 下有三份文件，可直接运行::

    python -m arff_compare examples/arff_compare/baseline.arff \
        examples/arff_compare/comparison.arff \
        --strata region,channel --class-attribute outcome \
        --weight-attribute survey_weight \
        --mapping examples/arff_compare/classes.json --format text

库调用
------

::

    from arff_compare import ClassMapping, compare_files

    report = compare_files(
        "baseline.arff", "comparison.arff",
        ["region", "channel"], "outcome",
        mapping=ClassMapping.from_file("classes.json"),
        weight_attribute="survey_weight",
    )

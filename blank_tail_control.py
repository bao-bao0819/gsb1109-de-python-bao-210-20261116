"""
XPT 库末尾空白的可行性分析。

XPT(v5/v6/v8/v9) 把观测顺序流出, 末尾用 ASCII 空格补齐到 80 字节块。
因此库末尾的空白既可能是块填充, 也可能是合法的全空字符观测。
本脚本在给定成员边界、记录宽度, 以及按成员给出的两项控制数
(笔数 = 观测数, 非空字段控制数 = 全表非空字段值总数) 的条件下:

1. 枚举每个成员所有可行的观测数;
2. 判定每个空尾行的确定性 (确定为观测 / 确定为填充 / 不确定);
3. 多成员的控制约束联合满足, 给出联合可行解与可行总笔数;
4. 无解时找出最小冲突控制项 (去掉它即可恢复可行的最小控制项集合)。

既不把空尾行全部删光 (n 取内容下界), 也不只挑一个碰巧读完的解释
(如 xport.v56 读到第一个全空行就停)。
"""

# Standard Library
import itertools
import math
import sys

# Xport Modules
import xport.v56 as v56
import xport.v89 as v89

BLANK = 0x20
BLOCK = 80
ANY = 'ANY'  # 标记: 任意 n >= 0 都可行 (记录宽度为 0 时)


def _module(bytestring):
    if bytestring.startswith(b'HEADER RECORD*******LIBV8'):
        return v89
    return v56


def member_regions(bytestring):
    """按成员头模式切出每个成员的观测字节区间 [obs_start, obs_end)。"""
    mod = _module(bytestring)
    if mod.Library.pattern.match(memoryview(bytestring)) is None:
        raise ValueError('不是合法的 XPT 文档')
    matches = list(mod.MemberHeader.pattern.finditer(memoryview(bytestring)))
    regions = []
    for k, m in enumerate(matches):
        obs_end = matches[k + 1].start(0) if k + 1 < len(matches) else len(bytestring)
        header = mod.MemberHeader.from_bytes(bytestring[m.start(0):m.end(0)])
        regions.append((header, m.end(0), obs_end))
    return regions


class MemberAnalysis:
    """单个成员的可行性分析结果。"""

    def __init__(self, header, region):
        self.name = header.name
        self.fields = [(ns.position, ns.length) for ns in header.values()]
        self.width = sum(length for _, length in self.fields)
        self.region = bytes(region)
        self.L = len(region)
        self.aligned = (self.L % BLOCK == 0)
        w = self.width
        if w == 0:
            # 每条观测 0 字节: 任何 n >= 0 都产生相同的 (空) 字节流。
            self.q = 0
            self.n_min = 0
            self.base = ANY
            self.f_star = 0
            self.n_blank_tail_rows = 0
            return
        # q = 区域内最后一个非空白字节之后的位置 (全空则为 0)。
        self.q = 0
        for i in range(self.L - 1, -1, -1):
            if region[i] != BLANK:
                self.q = i + 1
                break
        # 内容约束: 数据区必须覆盖全部非空白字节。
        self.n_min = math.ceil(self.q / w)
        # 尺寸约束: n*w <= L 且填充 L - n*w 必须在 [0, 79] (最小填充规则)。
        lo_size = math.ceil((self.L - (BLOCK - 1)) / w)
        self.lo = max(self.n_min, lo_size, 0)
        self.hi = self.L // w
        self.base = set(range(self.lo, self.hi + 1))
        # 非空字段控制数: 对任意可行 n 都相同 (空尾行贡献 0), 故只算一次。
        self.f_star = 0
        for row in range(self.n_min):
            for pos, length in self.fields:
                chunk = region[row * w + pos:row * w + pos + length]
                if any(b != BLANK for b in chunk):
                    self.f_star += 1
        # 字节上完整存在且全空的尾行 (行号从 1 起)。
        self.n_blank_tail_rows = self.hi

    def feasible(self, count=None, nonempty=None):
        """施加控制数后的可行观测数集合; 返回 set / ANY / 空集。"""
        ok = True
        if nonempty is not None and nonempty != self.f_star:
            ok = False
        if self.base is ANY:
            if not ok:
                return set()
            return ANY if count is None else {count}
        s = set(self.base)
        if count is not None:
            s &= {count}
        if not ok:
            s = set()
        return s

    def row_status(self, feasible):
        """逐空行判定: 返回 {行号: '确定为观测'/'确定为填充'/'不确定'}。"""
        status = {}
        if self.width == 0:
            return status
        if feasible is ANY:
            lo, hi = 0, math.inf
        elif feasible:
            lo, hi = min(feasible), max(feasible)
        else:
            return status
        for k in range(1, self.hi + 1):
            row_blank = k > self.n_min or self._row_is_blank(k)
            if not row_blank:
                continue
            if k <= lo:
                status[k] = '确定为观测'
            elif k > hi:
                status[k] = '确定为填充'
            else:
                status[k] = '不确定'
        return status

    def _row_is_blank(self, k):
        # 行号 k (从 1 起) 是否全空; 仅 k <= n_min 时需要逐字节判断。
        w = self.width
        chunk = self.region[(k - 1) * w:k * w]
        return all(b == BLANK for b in chunk)


def minimal_conflicts(analyses, controls):
    """
    在无解时找最小冲突控制项。

    controls: {成员名: {'count': c 或 None, 'nonempty': f 或 None}}
    返回 (结构冲突列表, 最小冲突控制项集合的列表)。
    控制项记为 (成员名, '笔数'|'非空字段控制数', 值)。
    """
    structural = [a.name for a in analyses
                  if not a.aligned or (a.base is not ANY and not a.base)]
    items = []
    for a in analyses:
        ctl = controls.get(a.name, {})
        if ctl.get('count') is not None:
            items.append((a.name, '笔数', ctl['count']))
        if ctl.get('nonempty') is not None:
            items.append((a.name, '非空字段控制数', ctl['nonempty']))

    def consistent(subset):
        for a in analyses:
            count = nonempty = None
            for name, kind, value in subset:
                if name != a.name:
                    continue
                if kind == '笔数':
                    count = value
                else:
                    nonempty = value
            if not a.feasible(count, nonempty):
                return False
        return True

    if consistent(items):
        return structural, []
    conflicts = []
    for size in range(1, len(items) + 1):
        for subset in itertools.combinations(items, size):
            if not consistent(subset):
                conflicts.append(set(subset))
        if conflicts:
            break  # 只保留最小规模
    return structural, conflicts


def report(bytestring, controls=None, title=''):
    controls = controls or {}
    analyses = []
    for header, start, end in member_regions(bytestring):
        analyses.append(MemberAnalysis(header, bytestring[start:end]))

    print(f'== {title} ==')
    joint = []
    for a in analyses:
        ctl = controls.get(a.name, {})
        s = a.feasible(ctl.get('count'), ctl.get('nonempty'))
        w = a.width
        print(f'成员 {a.name}: 记录宽度 w={w}, 观测区 L={a.L} 字节'
              + ('' if a.aligned else ' (长度非 80 的倍数, 文件被截断!)'))
        if w == 0:
            print(f'  无变量 (w=0): 任意 n>=0 的字节流完全相同; '
                  f'非空字段数恒为 0 (控制值 f*={a.f_star})')
        else:
            base = f'{{{min(a.base)}..{max(a.base)}}}' if a.base else '空集'
            print(f'  内容下界 n_min={a.n_min}, 尺寸区间 [{a.lo}, {a.hi}], '
                  f'基础可行集={base}, 非空字段数 f*={a.f_star}')
        c, f = ctl.get('count'), ctl.get('nonempty')
        if c is not None or f is not None:
            print(f'  控制项: 笔数={c}, 非空字段控制数={f}')
        if s is ANY:
            print('  可行观测数: 任意 n >= 0 (无控制数时无法确定)')
        elif s:
            print(f'  可行观测数: {sorted(s)}')
        else:
            print('  可行观测数: 无解')
        for k, v in a.row_status(s).items():
            print(f'    空行 {k}: {v}')
        joint.append(s)

    if joint and all(s for s in joint):
        if any(s is ANY for s in joint):
            print('联合解: 含无约束成员, 可行总笔数无界')
        else:
            combos = list(itertools.product(*(sorted(s) for s in joint)))
            totals = sorted({sum(c) for c in combos})
            print(f'联合解: {len(combos)} 种组合, 可行总笔数 = {totals}')
    else:
        structural, conflicts = minimal_conflicts(analyses, controls)
        for name in structural:
            print(f'结构冲突: 成员 {name} 的观测区本身无法合法解析 (与控制项无关)')
        for subset in conflicts:
            desc = ', '.join(f'{n} 的{k}={v}' for n, k, v in sorted(subset))
            print(f'最小冲突控制项: {{{desc}}}')
    print()
    return analyses


if __name__ == '__main__':
    # 情形 0: 随包示例 test/data/example.xpt —— 0 变量成员, w = 0。
    data = open('test/data/example.xpt', 'rb').read()
    report(data, title='example.xpt (成员 EMPTY, 0 变量), 无控制数')
    report(data, {'EMPTY': {'count': 4, 'nonempty': 0}},
           title='example.xpt, 控制 笔数=4 且 非空字段=0')
    report(data, {'EMPTY': {'count': 4, 'nonempty': 1}},
           title='example.xpt, 控制 笔数=4 但 非空字段=1 (冲突)')

    # 情形 1: 单成员, 尾部带 2 条全空字符观测。
    import xport
    ds1 = xport.Dataset(
        data={'A': ['A1', 'A2', 'A3', '', ''], 'B': ['x', 'y', 'z', '', '']},
        name='SITE01',
    )
    ds1['A'].width = ds1['B'].width = 4
    b1 = xport.v56.dumps(ds1)
    report(b1, title='SITE01: 3 条非空 + 2 条全空, 无控制数')
    report(b1, {'SITE01': {'count': 5, 'nonempty': 6}},
           title='SITE01, 控制 笔数=5 且 非空字段=6')
    report(b1, {'SITE01': {'count': 12, 'nonempty': 6}},
           title='SITE01, 控制 笔数=12 (超出尺寸上界, 冲突)')

    # 情形 2: 两个成员, 控制约束联合满足。
    ds2 = xport.Dataset(data={'C': ['hello', 'world']}, name='SITE02')
    ds2['C'].width = 8
    b2 = xport.v56.dumps(xport.Library(members=[ds1, ds2]))
    report(b2, title='SITE01+SITE02 联合, 无控制数')
    report(b2, {'SITE01': {'count': 5, 'nonempty': 6},
                'SITE02': {'count': 2, 'nonempty': 2}},
           title='SITE01+SITE02 联合, 两成员控制同时满足')

    # 情形 3: 两项控制各自都与文件矛盾 -> 两个最小冲突项。
    report(b1, {'SITE01': {'count': 1, 'nonempty': 99}},
           title='SITE01, 笔数=1 且 非空字段=99 (双双冲突)')

    # 情形 4: 观测区全空 —— 不允许把空尾行删光 (n=0 需 80 字节填充, 超界)。
    ds3 = xport.Dataset(data={'D': ['', '']}, name='BLANKS')
    ds3['D'].width = 8
    b3 = xport.v56.dumps(ds3)
    report(b3, title='BLANKS: 2 条全空观测, 无控制数 (n=0 不可行)')

    # 对照: 仓库现有读取器只给出"碰巧读完"的一种解释。
    for name, blob in [('SITE01', b1), ('BLANKS', b3)]:
        lib = v56.loads(blob)
        n = {k: len(v) for k, v in lib.items()}
        print(f'xport.v56 现有读取器对 {name} 的结果: {n} (单一解释, 非可行集)')

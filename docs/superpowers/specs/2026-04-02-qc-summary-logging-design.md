# QC Summary Logging Design

## Summary

QuantConnect 免费层回测日志容量很小，逐条调试日志容易被截断或去重，导致无法可靠判断关键策略路径是否执行。为避免后续继续被日志限制误导，本次设计在 QC 主链中增加低流量摘要计数日志：运行过程中只做轻量计数，回测结束时统一输出少量汇总行。

目标是让下次回测即使日志被限制，也能稳定看见这些关键行为是否发生：

- 是否出现过持股
- 是否生成过 `SELL_CALL`
- 是否生成过 `SELL_PUT`
- 是否发生过 `PUT_BLOCK`
- 是否跟踪过 assignment 股票
- 是否尝试过 immediate covered call
- 是否出现过股票买入/卖出成交

## Scope

本次只改 QuantConnect 主链日志，不改策略语义，不改参数，不同步到 `binbin_god` / parity 回放侧。

修改范围限定为：

- `quantconnect/strategy_init.py`
- `quantconnect/main.py`
- `quantconnect/signal_generation.py`
- `quantconnect/expiry.py`
- `quantconnect/strategy_mixin.py`
- 对应少量单元测试

## Design

### 1. Add Lightweight Debug Counters

在 `strategy_init.py` 初始化一个 `algo.debug_counters` 字典，用于记录关键事件的累计次数。默认计数器包括：

- `holdings_seen`
- `cc_signals`
- `sp_signals`
- `put_block`
- `sp_quality_block`
- `sp_stock_block`
- `sp_held_block`
- `assigned_stock_track`
- `assigned_repair_attempt`
- `assigned_repair_fail`
- `assigned_stock_exit`
- `immediate_cc`
- `stock_buy`
- `stock_sell`
- `no_suitable_options`

采用普通 `dict[str, int]` 即可，不引入新依赖，不增加复杂封装。

### 2. Increment Counters at Existing Log Points

在现有关键逻辑点保留现有日志行为，同时额外给对应计数器加一，保证：

- 即使逐条日志被截断，摘要统计仍然能反映真实执行情况
- 不改变当前排查时依赖的逐条日志格式

具体落点：

- `main.py`
  - 股票成交时累计 `stock_buy` / `stock_sell`
- `signal_generation.py`
  - 检测到持股时累计 `holdings_seen`
  - `SP_HELD_BLOCK`、`SP_STOCK_BLOCK`、`SP_QUALITY_BLOCK` 各自累计
- `strategy_mixin.py`
  - 记录 `CC_SIGNAL` 时累计 `cc_signals`
  - 记录 `SP_SIGNAL` 时累计 `sp_signals`
- `expiry.py`
  - `ASSIGNED_STOCK_TRACK`
  - `ASSIGNED_REPAIR_ATTEMPT`
  - `ASSIGNED_REPAIR_FAIL`
  - `ASSIGNED_STOCK_EXIT`
  - `IMMEDIATE_CC`
- `execution.py`
  - 本次不改；如果需要 `no_suitable_options` 计数，则在已有 `No suitable options for ...` 的调用点做一次轻量累计，避免引入额外摘要漏项

### 3. Emit Compact End-of-Algorithm Summary

在 `strategy_mixin.py:on_end_of_algorithm()` 里，在现有结果日志之后追加少量摘要行，建议控制为 2-3 行，格式固定，便于人工搜索：

示例：

```text
SUMMARY_FLOW: holdings_seen=3 cc_signals=4 sp_signals=32 put_block=22 no_suitable_options=3
SUMMARY_ASSIGNMENT: assigned_stock_track=1 immediate_cc=1 assigned_repair_attempt=2 assigned_repair_fail=0 assigned_stock_exit=0
SUMMARY_STOCK_FILLS: stock_buy=2 stock_sell=1 sp_quality_block=0 sp_stock_block=0 sp_held_block=0
```

这些行应尽量稳定，不频繁改 key 名，方便后续跨回测对比。

## Error Handling

- 若 `debug_counters` 缺失，读取时回退为 `0`，不要让摘要日志本身引发异常
- 若未来新增逻辑点但忘记初始化计数器，统一通过一个轻量 helper 自动补零并递增，避免 `KeyError`

## Testing

新增或更新单元测试，至少覆盖：

1. `init_state()` 或初始化路径会创建 `debug_counters`
2. 关键路径会递增计数器：
   - `CC_SIGNAL`
   - `SP_SIGNAL`
   - `ASSIGNED_STOCK_TRACK`
   - `STOCK_BUY` / `STOCK_SELL`
3. `on_end_of_algorithm()` 会输出固定格式的 `SUMMARY_*` 行
4. 当部分计数器未显式写入时，摘要仍能安全输出 `0`

## Trade-offs

### Why Not Remove Detailed Logs Entirely

这次不直接删掉逐条日志，因为它们在本地或会员环境下仍然有排查价值。摘要计数是为了在免费层下仍然保住关键行为可见性，而不是完全替代详细日志。

### Why Not Add a Full Debug Level System

增加 `debug_level` 会更灵活，但本次目标只是解决日志配额干扰，不值得把简单问题扩成一套新的配置系统。

## Success Criteria

下次 QC 回测结束后，即使中间详细日志被截断，也应该仍能在最终日志中看到：

- `SUMMARY_FLOW`
- `SUMMARY_ASSIGNMENT`
- `SUMMARY_STOCK_FILLS`

并能据此判断最新代码是否真的执行到了持股、covered call、assignment 和股票成交相关路径。

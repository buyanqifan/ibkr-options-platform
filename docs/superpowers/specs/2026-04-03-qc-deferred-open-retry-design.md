# QC Deferred Open Retry Design

## Summary

当前 QuantConnect 回测结果显示策略实际几乎没有成交 covered call，交易记录表现为大量 short put 加少数股票腿，而不是完整的 wheel。基于当前代码路径，最可能的缺口是 assignment 后或持股场景下生成的 `SELL_CALL` 在订阅新期权合约后因 `HasData=False` 被 `ORDER_DEFERRED`，随后没有真正进入重试流程，导致 call 腿直接丢失。

本次设计只解决这一条关键链路：给 deferred 的期权开仓请求增加轻量 pending 队列，并在后续 `rebalance()` 中优先重试，尤其是 deferred covered calls。

## Goal

确保以下场景不会因为首个 minute bar 尚未到达而永久丢失 `SELL_CALL`：

- put assignment 后的 immediate covered call
- 普通持股场景下的 covered call

同时不改变现有 put / roll / close 的主要语义，不引入新的大型订单状态机。

## Scope

本次只改 QuantConnect 主链，不同步到 `binbin_god / parity`。

修改范围限定为：

- `quantconnect/strategy_init.py`
- `quantconnect/execution.py`
- `quantconnect/strategy_mixin.py`
- `quantconnect/expiry.py`
- 对应 QC 单元测试

## Design

### 1. Add Pending Open Queue

在算法状态中新增一个轻量 `pending_open_orders` 字典或列表，用来保存 deferred 的开仓请求。每个请求至少包含：

- `option_symbol`
- `quantity`
- `theoretical_price`
- `signal`
- `selected`
- `target_right`
- `created_at`
- `attempt_count`

这份状态只用于“等待数据就绪后再次提交开仓单”，不承担 fill 后 metadata 存储职责。真正成交后仍然沿用现有 `pending_order_metadata`。

### 2. Queue Deferred Opens Instead of Dropping Them

调整 `safe_execute_option_order()` 语义：

- 当合约已有数据时：维持现状，直接下单并返回 `ticket`
- 当合约无数据时：
  - 保留 `ORDER_DEFERRED` 日志
  - 把本次开仓请求写入 `pending_open_orders`
  - 返回一个“已 defer”结果，而不是简单 `None`

为避免重复堆积，队列 key 应至少能唯一标识：

- 标的
- 期权方向
- expiry
- strike
- quantity

如果同一个 deferred open 已存在，则更新尝试时间或计数，而不是重复入队。

### 3. Retry Deferred Opens at Start of Rebalance

在 `rebalance()` 交易主流程开始阶段，加入一个 `retry_pending_open_orders()` 步骤：

- 在 position management 之后、生成新 signals 之前执行
- 优先重试 `SELL_CALL`
- 如果合约数据就绪，提交订单并移出 pending 队列
- 如果仍未就绪，保留在 pending 队列中

为避免无穷挂单，加入温和保护：

- 超过一定 age 或 attempt_count 的请求自动丢弃
- 丢弃时打印明确日志，例如 `ORDER_DEFERRED_EXPIRED`

默认保护建议：

- `max_attempts = 3`
- 或 `max_age_days = 3`

两者任选其一即可，本次优先选更简单的 `attempt_count`。

### 4. Keep Immediate CC on Same Path

`try_sell_cc_immediately()` 不做单独重试循环，也不做 sleep/poll。它仍然调用统一的 `execute_signal()`：

- 成功则当场下单
- 无数据则进入 `pending_open_orders`
- 后续由统一的 `rebalance()` retry 逻辑补单

这样 assignment 后 immediate CC 和普通 covered call 共享同一条补单路径，避免分叉逻辑。

### 5. Preserve Existing Fill Metadata Flow

本次不改 `pending_order_metadata` 语义：

- 只有真正拿到 `ticket` 时，才 enqueue open-order metadata
- `handle_order_event()` 继续只处理已提交订单的 fill

也就是说：

- `pending_open_orders` 负责“等待数据、重试提交”
- `pending_order_metadata` 负责“订单提交后、等待 fill 元数据落库”

两者职责分离。

## Error Handling

- 如果 `pending_open_orders` 缺失，重试函数安全返回
- 如果 deferred 请求缺字段，丢弃并打一次警告日志
- 如果超过最大尝试次数，丢弃并记录 `ORDER_DEFERRED_EXPIRED`
- 重试过程中如果 `signal` 已不再适用，不额外重算信号；本次按“重试原始请求”处理，减少复杂度

## Testing

新增或更新测试，至少覆盖：

1. `safe_execute_option_order()` 在无数据时会把请求写入 `pending_open_orders`
2. `rebalance()` 开始时会优先重试 deferred covered calls
3. 数据就绪后 deferred open 会真正下单并从队列移除
4. 多次重试后仍无数据会过期移除
5. immediate CC 的 defer 路径会进入同一队列，而不是直接丢失

## Trade-offs

### Why Not Retry Inside `try_sell_cc_immediately()`

同步重试更依赖 QC 的数据到达时序，而且会让 assignment 路径和普通 CC 路径分叉。统一走 pending 队列更稳定，也更容易测试。

### Why Not Build a Full Open-Order State Machine

完整状态机会更通用，但当前最痛点非常集中：covered call 因无数据被 defer 后没有补单。本次只解决这个关键缺口，避免把问题扩成大重构。

## Success Criteria

下次回测如果出现股票腿：

- 交易记录中应能开始看到真实 `call` 订单 / 成交
- 不再只剩 `put + stock`
- assignment 后的股票更有机会被 wheel 回 covered call 路径

即使日志继续被免费层截断，只要交易记录中 `call` 笔数从 `0` 变成非零，就说明这条优化生效了。

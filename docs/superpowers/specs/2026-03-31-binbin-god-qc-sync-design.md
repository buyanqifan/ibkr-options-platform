# Binbin God QC Sync Design

## Goal

将 `quantconnect` 当前版本里已经验证过的策略行为、参数配置和 parity sizing 规则同步到自研引擎的 `binbin_god`，在尽量不破坏现有接口和 UI 的前提下，让两套实现的回测行为更接近。

## Scope

本次同步覆盖以下内容：

1. 将 `quantconnect` 新增的参数同步到自研 `binbin_god` 和 `qc_parity` 配置层。
2. 将 `quantconnect` 新增的策略行为同步到自研 `binbin_god`：
   - `SP` 开仓前 cooldown 拦截
   - `SP` 开仓前 stock inventory cap 拦截
   - `CC` repair mode
   - `short put` defensive roll
   - put assignment / 大亏后的 cooldown 触发
3. 将 `quantconnect` 当前版本的 symbol 风险折扣、波动率加权单票 cap、库存 cap 同步到 `qc_parity.py`，用于 parity 模式下的仓位计算。
4. 在必要时做最小范围 UI 调整，但 UI 不是主目标，只有在参数透传或页面兼容被本次同步直接影响时才修改。

## Non-Goals

本次不做以下内容：

1. 不把自研 `binbin_god` 完全重写成 `quantconnect` 的模块化目录结构。
2. 不重构无关逻辑，不改无关页面，不做风格化 UI 调整。
3. 不试图一次性消除所有自研引擎与 QC 的差异，只同步当前明确新增且会影响交易行为的部分。

## Current Gaps

相对于 `quantconnect` 当前版本，自研 `binbin_god` 目前缺少这些关键能力：

1. `repair_call_*` 参数和对应的 CC 修复逻辑。
2. `defensive_put_roll_*` 参数和独立于 ML / 传统 roll 规则的 defensive roll。
3. `assignment_cooldown_days`、`large_loss_cooldown_*` 以及 symbol 级 cooldown 状态。
4. `volatility_cap_*`、`dynamic_symbol_risk_enabled`、`symbol_*_sensitivity`、`symbol_assignment_base_cap`、`stock_inventory_*` 等新版仓位风控参数。
5. `qc_parity.py` 仍然使用旧版 QC sizing 规则，缺少 symbol 风险折扣和库存上限。

## Design

### 1. Parameter Sync

在 `core/backtesting/strategies/binbin_god.py` 初始化阶段补齐所有新版 QC 参数，并设置与 QC 一致的默认值。参数命名优先直接沿用 QC 版本，避免未来再次同步时出现映射层。

在 `core/backtesting/qc_parity.py` 的 `QC_BINBIN_DEFAULTS` 和 `BinbinGodParityConfig` 中同步同一批参数，使 parity 模式能够直接消费这些配置。

### 2. Strategy State Sync

在自研 `BinbinGodStrategy` 中新增最小状态：

1. `symbol_cooldowns`
2. cooldown 读写辅助方法
3. 与 repair / defensive roll 判断所需的轻量辅助方法

状态设计保持局部化，避免把 QC 的 portfolio wrapper 或 metadata 机制原样搬入自研引擎。

### 3. Signal Generation Sync

在自研策略生成 `SP` 信号时增加两层拦截：

1. symbol cooldown
2. stock inventory cap

在生成 `CC` 信号时加入 repair call 模式：

1. 当 underlying 相对 cost basis 的回撤超过阈值时，强制提高最小 call delta。
2. 限制 DTE 落入 repair call 范围。
3. 设置最小 strike，避免为了收权利金卖出过低执行价。

### 4. Position Management Sync

在自研持仓管理中，把 short put 的处理顺序调整为：

1. 先判断 defensive put roll
2. 再走现有 ML roll
3. 最后走已有传统 roll / close 规则

如果 defensive roll、put 大亏 roll/close、put assignment 发生，则按 QC 规则设置 symbol cooldown。

### 5. Parity Sizing Sync

在 `core/backtesting/qc_parity.py` 中加入和 QC 当前版本一致的新版 sizing 逻辑：

1. 波动率加权单票 put cap
2. symbol 状态风险折扣
3. 动态 stock inventory cap
4. 这些限制进入 `calculate_put_quantity_qc()` 的最终 `min()` 约束集合

这样 parity 模式不只是“参数名对齐”，而是仓位限制逻辑也更贴近 QC。

### 6. UI Compatibility

默认不改 UI。

如果本次新增参数需要在现有页面中暴露，或者页面读取配置时因为缺少字段导致异常，则只做最小兼容修改。修改原则是：

1. 只补字段，不重设计页面
2. 不改变用户现有交互路径
3. 只有当同步后的策略无法被前端正确配置时才改

## Files

预计主要修改文件：

1. `core/backtesting/strategies/binbin_god.py`
2. `core/backtesting/qc_parity.py`
3. 与 `binbin_god` 参数或测试直接相关的现有测试文件
4. 必要时补充极少量 UI / 页面文件

## Testing

至少覆盖以下验证：

1. 参数默认值和解析是否与 QC 对齐
2. cooldown 是否阻止新 `SP`
3. inventory cap 是否阻止 stock-heavy symbol 继续卖 put
4. repair call 是否改变 delta / strike / DTE
5. defensive put roll 是否在 ITM / 大亏时优先触发
6. parity sizing 是否体现 symbol 风险折扣和库存上限

## Risks

1. 自研引擎与 QC 的数据结构不同，不能逐行搬运逻辑，需要做语义级映射。
2. `binbin_god.py` 文件较大，改动要控制在局部，避免把无关行为带坏。
3. parity helper 与主策略可能出现“双重风控”或配置漂移，因此需要明确哪些规则只在 parity 模式下生效，哪些是全局策略行为。

## Acceptance Criteria

满足以下条件视为完成：

1. 自研 `binbin_god` 支持当前 QC 新增的关键参数。
2. 自研 `binbin_god` 具备 cooldown、repair call、defensive put roll、inventory cap 等关键行为。
3. parity 模式下的 put sizing 规则与当前 `quantconnect` 版本显著更接近。
4. 现有测试通过，新增测试覆盖本次同步的核心行为。
5. 若 UI 有改动，改动仅限本次同步所必需的兼容层。

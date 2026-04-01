# IBKR Options Trading Platform

基于盈透证券 (Interactive Brokers) API 的美股期权交易平台，支持实时行情、股票筛选、多种期权策略回测、ML智能Delta优化，提供现代化Web界面。

## 功能特性

### 核心功能

- **实时行情** - 通过 IBKR API 获取美股实时报价和历史 K 线数据
- **期权链查看** - 完整期权链、Greeks (Delta/Gamma/Theta/Vega)、IV Smile 曲线
- **股票筛选器** - 按财务指标 (PE、市值)、期权 IV Rank、技术指标筛选股票
- **策略回测** - 支持 9 种期权策略回测，完整的收益指标和可视化
- **Binbin God 实盘模拟盘** - 独立 Live 页面、后台 worker、IBKR 模拟账户自动交易、恢复保护和事件审计
- **ML Delta优化** - 基于强化学习的智能Delta选择，适应不同市场状态
- **多语言支持** - 支持中英文切换
- **Docker 部署** - 一键部署（含 IB Gateway 容器）

## 最近更新

### 2026-04-01

- 新增 `Binbin God Live` 模拟盘实盘模块，提供独立网页、后台常驻 worker、订单审计和重启恢复保护。
- 新增 IBKR 下单适配层与实盘持久化模型，支持通过网页启停策略、查看恢复结果、订单/成交/持仓和风险状态。
- 修复数据库连接与 SQLite schema 兼容问题，确保本地 `DB_PATH` 切换和旧库升级时行为更稳定。

### 支持的回测策略

| 策略 | 说明 | ML Delta支持 |
|------|------|-------------|
| Sell Put | 卖出看跌期权（Cash Secured Put） | ✅ |
| Covered Call | 备兑看涨期权 | ✅ |
| Iron Condor | 铁鹰策略 | ✅ |
| Bull Put Spread | 牛市看跌价差 | ✅ |
| Bear Call Spread | 熊市看涨价差 | ✅ |
| Short Straddle | 卖出跨式（ATM） | ❌ |
| Short Strangle | 卖出宽跨式 | ✅ |
| Wheel Strategy | 轮动策略（SP + CC 循环） | ❌ |
| Binbin God | 智能选股 + Wheel 策略 | ✅ |

### ML Delta 优化功能

机器学习驱动的Delta智能选择系统：

- **市场状态识别** - 自动判断 Bull/Bear/Neutral/High_Vol 四种市场状态
- **强化学习** - Q-Learning 算法持续优化Delta选择
- **预训练机制** - 使用历史数据预训练，快速获得有效策略
- **自适应策略** - 根据信任度自动调整ML与传统策略的权重

## 技术栈

| 层级 | 技术 |
|------|------|
| **后端** | Python 3.11, ib_insync, Flask |
| **前端** | Plotly Dash, Dash Bootstrap Components, AG Grid |
| **期权定价** | Black-Scholes (scipy) |
| **机器学习** | NumPy, Pandas, Q-Learning |
| **数据库** | SQLite + SQLAlchemy |
| **部署** | Docker + Docker Compose |
| **IBKR连接** | [extrange/ibkr-docker](https://github.com/extrange/ibkr-docker) |

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Browser                             │
│                    (Dash Frontend)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dash Application                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │Dashboard│ │MarketData│ │Screener│ │Backtester│           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │Options Chain│ │  Settings   │ │ Binbin God  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────────────┐                                     │
│  │ Binbin God Live     │                                     │
│  └─────────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Services                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ IBKR Client  │  │Backtest Engine│ │ ML Optimizer │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │Position Mgr  │  │  Screener    │  │ Vol Predictor│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    IB Gateway / TWS                          │
│                  (IBKR API Connection)                       │
└─────────────────────────────────────────────────────────────┘
```

## 前提条件

1. **盈透证券账户** - 需要 IBKR 账户（支持 Paper Trading 模拟账户）
2. **API 权限** - 在 IBKR 账户管理页面启用 API 访问

## 快速开始

### 方式一：Docker 部署（推荐）

使用第三方 IB Gateway Docker 镜像，自动启动 IB Gateway 容器，无需手动安装 TWS。

```bash
# 克隆仓库
git clone https://github.com/buyanqifan/ibkr-options-platform.git
cd ibkr-options-platform

# 创建配置文件
cp .env.example .env

# 编辑 .env，填入 IBKR 凭据
nano .env
```

**.env 文件内容：**
```env
# IBKR Credentials (for IB Gateway Docker)
USERNAME=你的 IBKR 用户名
PASSWORD='你的 IBKR 密码'  # 如果密码包含特殊字符，用单引号包裹

# IB Gateway Configuration
GATEWAY_OR_TWS=gateway
IBC_TradingMode=paper
IBC_ReadOnlyApi='no'
IBC_ExistingSessionDetectedAction=primaryonly
IBC_AutoRestart='yes'

# App Settings
IBKR_HOST=ibgateway
IBKR_PORT=8888
IBKR_CLIENT_ID=1
IBKR_TRADING_MODE=paper
APP_HOST=0.0.0.0
APP_PORT=8050
APP_DEBUG=false
DB_PATH=data/trading.db
LOG_LEVEL=INFO
```

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down
```

浏览器打开 http://localhost:8050

---

### 方式二：本地运行

#### 1. 安装 TWS 或 IB Gateway

从盈透官网下载并安装：
- [TWS (Trader Workstation)](https://www.interactivebrokers.com/en/trading/tws.php) - 完整交易客户端
- [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) - 轻量版，无 GUI

#### 2. 登录并配置 API

1. 启动 TWS 或 IB Gateway，用 IBKR 账号登录
2. 配置 API 连接：
   - **TWS**: `Edit` -> `Global Configuration` -> `API` -> `Settings`
   - **IB Gateway**: 启动时自动配置
3. 确保以下选项已启用：
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Allow connections from localhost only
4. 记住端口号：
   - Paper Trading: `7497` (TWS) 或 `4002` (Gateway)
   - Live Trading: `7496` (TWS) 或 `4001` (Gateway)

#### 3. 安装依赖并启动

```bash
# 克隆仓库
git clone https://github.com/buyanqifan/ibkr-options-platform.git
cd ibkr-options-platform

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 创建配置文件
cp .env.example .env

# 编辑配置
# IBKR_HOST=127.0.0.1
# IBKR_PORT=7497  # TWS Paper Trading
# IBKR_TRADING_MODE=paper

# 启动应用
PYTHONPATH=. python -m app.main
```

浏览器打开 http://localhost:8050

## 页面功能

| 页面 | 功能说明 |
|------|---------|
| **Dashboard** | 连接状态、账户摘要（净值/现金/P&L）、当前持仓列表 |
| **Market Data** | 股票搜索、K 线图（支持 MA20/50/200）、实时报价 |
| **Screener** | 设置筛选条件（PE、市值、IV Rank 等），查看评分排名 |
| **Options Chain** | 期权链表格、Greeks 数据、IV Smile 曲线图 |
| **Backtester** | 策略选择、参数配置、ML Delta优化、P&L 曲线、月度热力图 |
| **Binbin God Live** | Binbin God 模拟盘实盘控制台，支持参数配置、启停、恢复结果、风险状态、订单/成交/持仓/事件展示 |
| **Settings** | IBKR 连接配置、手动连接/断开、缓存管理、语言切换 |

## 回测器详细说明

### 参数配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| Strategy | 选择策略 | Sell Put |
| Symbol | 股票代码 | NVDA |
| Date Range | 回测日期范围 | 2025-01-01 至今 |
| Initial Capital | 初始资金 | $150,000 |
| Max Leverage | 最大杠杆 | 1.0 |
| DTE Range | 到期天数范围 | 30-45 天 |
| Target Delta | 目标 Delta | 0.30 |
| Profit Target | 止盈比例 | 50% |
| Stop Loss | 止损比例 | 200% |
| Max Positions | 最大持仓数 | 5 |
| ML Delta Optimization | ML Delta优化 | 关闭 |
| ML Adoption Rate | ML采纳率 | 0.5 |

### ML Delta 优化

启用 ML Delta 优化后，系统将：

1. **预训练阶段** - 使用前30%的历史数据模拟交易，建立初始Q-table
2. **市场状态识别** - 实时判断当前市场状态
3. **智能Delta选择** - 根据市场状态和历史表现选择最优Delta

**市场状态判断规则：**

| 状态 | 条件 | 推荐Delta |
|------|------|-----------|
| Bull | 5日动量 > 5% 且 20日动量 > 10% | Put: 0.30, Call: 0.40 |
| Bear | 5日动量 < -5% 且 20日动量 < -10% | Put: 0.20, Call: 0.30 |
| High Vol | 20日波动率 > 30% | Put: 0.25, Call: 0.20 |
| Neutral | 其他情况 | Put: 0.30, Call: 0.30 |

### 输出指标

| 指标 | 说明 |
|------|------|
| Total Return | 总收益率 |
| Annual Return | 年化收益率 |
| Win Rate | 胜率 |
| Sharpe Ratio | 夏普比率 |
| Max Drawdown | 最大回撤 |
| Profit Factor | 盈亏比 |
| Sortino Ratio | 索提诺比率 |

## Binbin God 策略

智能选股 + Wheel 策略组合，自动从 MAG7 中选择最优股票进行交易。

### 选股评分体系

| 因素 | 权重 | 说明 |
|------|------|------|
| IV Rank | 35% | 波动率排名，高IV = 更高权利金 |
| Technical | 25% | RSI + 均线位置 |
| Momentum | 20% | 价格动量 |
| PE Score | 20% | 估值因素 |

### 运作流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 1: Sell Put                       │
│  • 从 MAG7 中评分选择最优股票                                 │
│  • 卖出 OTM Put 收取权利金                                    │
│  • 到期未被行权 → 继续卖出 Put                                │
│  • 被行权 → 以 Strike 价格买入股票，进入 Phase 2              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Phase 2: Covered Call                    │
│  • 持有股票时卖出 OTM Call                                    │
│  • 到期未被行权 → 继续卖出 Call                               │
│  • 被行权 → 以 Strike 价格卖出股票，返回 Phase 1              │
│  • ML Delta 优化：股价低于成本时自动降低 Call Delta           │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
ibkr-options-platform/
├── app/                        # Dash Web 应用
│   ├── pages/                  # 页面模块
│   │   ├── dashboard.py        # 仪表盘
│   │   ├── market_data.py      # 行情页面
│   │   ├── screener.py         # 股票筛选器
│   │   ├── options_chain.py    # 期权链
│   │   ├── backtester.py       # 回测器
│   │   ├── binbin_god.py       # Binbin God 页面
│   │   └── settings.py         # 设置页面
│   ├── components/             # 可复用组件
│   │   ├── charts.py           # 图表组件
│   │   ├── tables.py           # 表格组件
│   │   ├── monitoring.py       # 监控组件
│   │   └── connection_status.py
│   ├── main.py                 # 应用入口
│   ├── layout.py               # 布局定义
│   ├── services.py             # 服务初始化
│   └── i18n.py                 # 国际化
├── core/
│   ├── ibkr/                   # IBKR 连接和数据
│   │   ├── connection.py       # 连接管理
│   │   ├── data_client.py      # 数据客户端
│   │   └── event_bridge.py     # 事件桥接
│   ├── market_data/            # 行情数据缓存
│   ├── screener/               # 股票筛选器
│   │   ├── screener.py         # 筛选器主逻辑
│   │   ├── criteria.py         # 筛选条件
│   │   ├── filters.py          # 过滤器
│   │   └── ranker.py           # 排序器
│   └── backtesting/            # 回测引擎
│       ├── engine.py           # 回测引擎
│       ├── simulator.py        # 交易模拟器
│       ├── metrics.py          # 性能指标
│       ├── pricing.py          # 期权定价
│       ├── position_manager.py # 仓位管理
│       ├── cost_model.py       # 交易成本模型
│       └── strategies/         # 策略实现
│           ├── base.py         # 策略基类
│           ├── sell_put.py     # 卖Put策略
│           ├── covered_call.py # 备兑Call策略
│           ├── iron_condor.py  # 铁鹰策略
│           ├── spreads.py      # 价差策略
│           ├── straddle.py     # 跨式策略
│           ├── wheel.py        # Wheel策略
│           └── binbin_god.py   # Binbin God策略
├── core/ml/                    # 机器学习模块
│   ├── delta_optimizer.py      # Delta优化器
│   ├── delta_strategy_integration.py  # 策略集成
│   ├── features/               # 特征工程
│   │   ├── volatility.py       # 波动率特征
│   │   └── technical.py        # 技术指标特征
│   └── models/                 # ML模型
├── models/                     # SQLAlchemy 数据模型
├── config/                     # 配置文件
│   ├── settings.py             # 应用设置
│   └── strategies.py           # 策略配置
├── utils/                      # 工具函数
│   ├── logger.py               # 日志工具
│   ├── date_utils.py           # 日期工具
│   └── rate_limiter.py         # 速率限制
├── data/                       # 数据目录
│   └── models/                 # ML模型存储
├── tests/                      # 测试用例
├── Dockerfile
├── docker-compose.yml
├── docker-compose.ssl.yml      # SSL配置
├── requirements.txt
└── .env.example
```

## ML Delta 优化原理

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    DeltaOptimizerML                          │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Q-Table      │  │ Performance  │  │ Market Regime│       │
│  │ 状态-动作价值 │  │ History      │  │ Patterns     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│          │                │                  │               │
│          └────────────────┼──────────────────┘               │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            optimize_delta()                          │    │
│  │  1. 提取市场特征 (波动率, 动量, etc)                  │    │
│  │  2. 评分候选Delta (0.05 - 0.40)                      │    │
│  │  3. 选择最优Delta + 探索机制                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 评分公式

```
总分 = 历史表现(30%) + 市场状态适应性(25%) + 风险调整收益(25%) + 持仓优化(20%)
```

### 在线学习

每次交易完成后更新Q-table：

```python
Q(state, action) = Q(state, action) + α × (reward - Q(state, action))
```

其中：
- `state` = symbol + market_regime + price_level
- `action` = delta值
- `reward` = 实际P&L
- `α` = 学习率 (默认0.01)

### 性能特点

| 指标 | 要求 |
|------|------|
| CPU | 极低（仅字典查找） |
| 内存 | <10MB |
| 延迟 | <1ms |
| GPU | 不需要 |

## 常见问题

### Q: 连接失败怎么办？

1. 确认 TWS/IB Gateway 已启动并登录
2. 检查端口号是否正确：
   - Paper Trading: `7497` (TWS) / `4002` (Gateway)
   - Live Trading: `7496` (TWS) / `4001` (Gateway)
3. 确认 TWS/Gateway 中 API 已启用且允许 localhost 连接
4. 检查防火墙设置
5. 使用 Docker 部署时，检查 `.env` 文件中的凭据是否正确

### Q: 没有 IBKR 账户能用吗？

可以！
- **回测功能** (Backtester) 不需要连接 IBKR，可使用合成历史数据
- 在回测器页面勾选 "Use Random Synthetic Data"
- 可在 [IBKR 官网](https://www.interactivebrokers.com/) 申请免费的 Paper Trading 模拟账户

### Q: Paper Trading 和 Live Trading 有什么区别？

| 模式 | 说明 |
|------|------|
| Paper Trading | 模拟账户，虚拟资金，零风险，用于测试和学习 |
| Live Trading | 真实账户，真金白银，请谨慎操作 |

### Q: 市场数据有延迟吗？

- 默认获取的是延迟数据（15-20 分钟）
- 实时数据需要在 IBKR 账户中订阅市场数据服务

### Q: ML Delta 优化需要多少历史数据？

- 最少需要 60 根 K 线才能进行预训练
- 建议使用至少 1 年的历史数据获得更好效果
- 数据越多，模型学习效果越好

### Q: 如何调整 ML 的信任程度？

使用 `ML Adoption Rate` 参数：
- `0.0` - 完全使用传统策略
- `0.5` - 平衡传统和ML（推荐）
- `1.0` - 完全信任ML结果

## 开发指南

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/buyanqifan/ibkr-options-platform.git
cd ibkr-options-platform

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装开发依赖
pip install -r requirements.txt

# 运行测试
pytest tests/
```

### 添加新策略

1. 在 `core/backtesting/strategies/` 创建新策略文件
2. 继承 `BaseStrategy` 类
3. 实现 `name` 属性和 `generate_signals()` 方法
4. 在 `core/backtesting/engine.py` 的 `STRATEGY_MAP` 中注册

### 代码风格

- 使用 Python 3.11+ 语法
- 遵循 PEP 8 编码规范
- 提交信息使用英文，遵循 Conventional Commits

## 更新日志

### v1.2.0 (2026-03)
- 新增 ML Delta 优化功能
- 新增 Binbin God 智能选股策略
- 新增 ML 预训练机制
- 优化回测性能
- 修复多个已知问题

### v1.1.0 (2025-12)
- 新增多语言支持（中英文）
- 新增合成数据功能
- 优化仓位管理系统
- 新增交易成本模型

### v1.0.0 (2025-09)
- 初始版本发布
- 支持 7 种期权策略回测
- 实现完整的 Web 界面
- Docker 部署支持

## 免责声明

本项目仅供学习和研究用途。期权交易存在风险，请在充分了解风险的情况下谨慎操作。作者不对使用本软件造成的任何损失负责。

## License

MIT

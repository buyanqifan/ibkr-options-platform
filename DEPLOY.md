# IBKR Docker 部署说明

## 🚨 重要提示

**首次部署必须通过 noVNC 手动配置 API 访问权限！**

## 🔧 服务器配置步骤

### 1. 确保 .env 文件存在（服务器环境）

```bash
cd /opt/ibkr-options-platform

# 创建或编辑 .env 文件
cat > .env << EOF
# IBKR 账户信息（必填）
USERNAME=您的 IBKR 用户名
PASSWORD='您的 IBKR 密码'  # 如果密码包含 $, /, \ 等特殊字符，用单引号包裹

# 交易模式：paper（模拟）或 live（真实）
IBC_TradingMode=paper
EOF
```

### 2. 启动容器

```bash
cd /opt/ibkr-options-platform
docker compose down
docker compose up -d

# 等待 IB Gateway 完全启动（约 2-3 分钟）
sleep 180
```

### 2.1 Binbin God Live worker 说明

`Binbin God Live` 现在作为独立后台 worker 随 `docker compose up -d` 一起启动，服务名为：

- `binbin_god_live_worker`

这个 worker 会连接 IBKR 模拟账户、读取数据库中的实盘参数和控制命令，并在重启后先执行恢复模式，再默认停在 `stopped`，等待网页手动启动策略。

### 3. 通过 noVNC 配置 API 访问（必须！）

#### 方式一：浏览器访问 noVNC
```
http://你的服务器IP:6080
```

#### 方式二：SSH 隧道（推荐，更安全）
```bash
# 在本地终端执行
ssh -L 6080:localhost:6080 user@你的服务器IP

# 然后在浏览器打开 http://localhost:6080
```

#### 在 noVNC 界面中的操作：

1. **等待自动登录完成**（约 30-60 秒）
2. 点击菜单栏：**File → Global Configuration...**
3. 展开左侧：**API → Settings**
4. 勾选以下选项：
   - ✅ **Enable ActiveX and Socket Clients**
   - ✅ **Allow connections from localhost only** （取消勾选！因为我们在不同容器）
   - 设置 **Socket port**: `4002`（应该已自动设置）
5. 点击 **OK** 保存
6. 如果有弹出窗口询问是否接受连接，点击 **Yes**

### 4. 验证连接

```bash
# 查看容器日志，确认 API 配置
docker compose logs ibgateway | grep -i "api\|socket"

# 检查端口是否正常监听
docker exec ibkr-options-platform_ibgateway_1 netstat -tlnp | grep 4002

# 测试从 app 容器连接
docker exec ibkr-options-platform_app_1 python3 -c "
from ib_insync import IB
ib = IB()
try:
    ib.connect('ibgateway', 8888, clientId=1, timeout=5)
    print('✅ 连接成功！')
    ib.disconnect()
except Exception as e:
    print(f'❌ 连接失败：{e}')
"
```

### 5. 访问期权交易平台

```
http://你的服务器IP:8050/settings
```

在 Settings 页面点击 **Connect** 按钮测试连接。

然后可以进入：

```
http://你的服务器IP:8050/binbin-god-live
```

在 `Binbin God Live` 页面中：

1. 确认顶部状态栏里 `Trading Mode` 为 `paper`
2. 确认 worker 已经有心跳、恢复结果正常
3. 保存/加载/应用实盘参数
4. 点击 `Start` 后再开始自动交易

## 🔐 阿里云安全组配置

如果使用 `6080:6080` 和 `8050:8050` 的端口绑定（非 127.0.0.1），需要在阿里云 ECS 安全组开放：

- TCP 6080（noVNC 访问）
- TCP 8050（应用访问）
- TCP 8888（可选，API 直接访问）

**建议**：使用 SSH 隧道访问 noVNC，这样更安全，不需要开放 6080 端口。

## 📝 docker-compose.yml 配置说明

```yaml
services:
  ibgateway:
    image: ghcr.io/extrange/ibkr:stable  # 使用 stable 版本更稳定
    ports:
      - "6080:6080"  # noVNC（可改为 "127.0.0.1:6080:6080" 仅本地访问）
      - "8888:8888"  # API 端口（自动转发到 4002）
    ulimits:
      nofile: 10000  # 必须设置，避免文件描述符错误
    environment:
      USERNAME: ${USERNAME}
      PASSWORD: '${PASSWORD}'
      GATEWAY_OR_TWS: gateway  # gateway 占用资源更少
      IBC_TradingMode: paper   # paper 或 live
      IBC_ReadOnlyApi: 'no'    # 允许交易操作
      IBC_ExistingSessionDetectedAction: primaryonly  # 自动处理会话冲突
      IBC_AutoRestart: 'yes'   # 自动重启
  app:
    build: .
    restart: unless-stopped
    ports:
      - "8050:8050"
    environment:
      IBKR_HOST: ibgateway
      IBKR_PORT: "8888"
      IBKR_CLIENT_ID: "1"
      IBKR_TRADING_MODE: ${IBKR_TRADING_MODE:-paper}
      DB_PATH: "data/trading.db"
  binbin_god_live_worker:
    build: .
    restart: unless-stopped
    command: ["python", "-m", "core.live_trading.binbin_god.runner"]
    environment:
      IBKR_HOST: ibgateway
      IBKR_PORT: "8888"
      IBKR_CLIENT_ID: "1"
      IBKR_TRADING_MODE: ${IBKR_TRADING_MODE:-paper}
      DB_PATH: "data/trading.db"
```

## ⚠️ 常见问题

### Q: 连接后立即断开
**A**: 需要在 noVNC 界面手动启用 API 访问（见步骤 3）

### Q: 无法访问 noVNC (6080 端口)
**A**: 
- 检查阿里云安全组是否开放 6080 端口
- 或使用 SSH 隧道：`ssh -L 6080:localhost:6080 user@server`

### Q: 容器启动后很快退出
**A**: 
- 检查 .env 文件中的用户名密码是否正确
- 查看日志：`docker compose logs ibgateway`
- 确保 ulimit nofile=10000 已设置

### Q: TWS_SETTINGS_PATH 是什么？
**A**: 用于持久化保存 TWS 设置，例如：
```yaml
environment:
  TWS_SETTINGS_PATH: /settings
volumes:
  - ./ibkr-settings:/settings:rw
```

## 🔄 重启容器

```bash
# 完全重启
docker compose restart

# 只重启 IB Gateway
docker compose restart ibgateway

# 只重启应用
docker compose restart app

# 只重启 Binbin God Live worker
docker compose restart binbin_god_live_worker
```

## 🤖 Binbin God Live 部署 / 重启步骤

### 首次部署

```bash
cd /opt/ibkr-options-platform
git pull origin main
docker compose build app binbin_god_live_worker
docker compose up -d ibgateway app binbin_god_live_worker
docker compose logs -f ibgateway
```

等待 IB Gateway 完全启动并完成 API 设置后：

```bash
docker compose logs -f binbin_god_live_worker
```

确认 worker 没有持续报错，然后打开：

```text
http://你的服务器IP:8050/binbin-god-live
```

### 日常重启

```bash
cd /opt/ibkr-options-platform
docker compose restart app binbin_god_live_worker
```

如果 IB Gateway 也需要一起重启：

```bash
docker compose restart ibgateway app binbin_god_live_worker
```

### 升级代码后重启

```bash
cd /opt/ibkr-options-platform
git pull origin main
docker compose build app binbin_god_live_worker
docker compose up -d app binbin_god_live_worker
```

升级后建议检查：

```bash
docker compose logs --tail=100 binbin_god_live_worker
docker compose logs --tail=100 app
```

然后到 `Binbin God Live` 页面确认：

1. 恢复结果不是 `blocked`
2. 恢复时间是本次重启之后
3. 持仓数、挂单数、异常数符合预期
4. 策略状态默认是 `stopped`
5. 人工确认无误后再点击 `Start`

## 📊 日志查看

```bash
# 查看所有服务日志
docker compose logs -f

# 只看 IB Gateway 日志
docker compose logs -f ibgateway

# 只看应用日志
docker compose logs -f app

# 只看 Binbin God Live worker 日志
docker compose logs -f binbin_god_live_worker

# 搜索 API 相关错误
docker compose logs ibgateway | grep -i "api\|error\|reject"
```

## 🎯 下一步

配置完成后，访问 http://你的服务器IP:8050 开始使用期权交易平台！

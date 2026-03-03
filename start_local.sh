#!/bin/bash
# 本地开发启动脚本

set -e

echo "🚀 启动本地开发环境..."

# 设置本地开发环境变量
export $(grep -v '^#' .env.local | xargs)

echo "📋 当前配置:"
echo "  IBKR_HOST: $IBKR_HOST"
echo "  IBKR_PORT: $IBKR_PORT"
echo "  APP_PORT: $APP_PORT"
echo "  DEBUG MODE: $APP_DEBUG"

# 检查IBKR服务是否运行
echo "🔍 检查IBKR服务..."
if nc -z $IBKR_HOST $IBKR_PORT; then
    echo "✅ IBKR服务在 $IBKR_HOST:$IBKR_PORT 可用"
else
    echo "⚠️  IBKR服务不可用 - 请确保TWS或IB Gateway已启动"
    echo "💡 启动步骤:"
    echo "   1. 启动TWS或IB Gateway客户端"
    echo "   2. 确保API设置已启用"
    echo "   3. 允许localhost连接"
    echo "   4. 使用端口 4002 (纸币交易) 或 4001 (实盘交易)"
fi

# 启动应用
echo "🎮 启动Web应用..."
python3 -m app.main
#!/bin/bash
# SSH隧道脚本 - 用于连接远程服务器进行开发调试

SERVER_IP="8.148.154.206"
LOCAL_PORT=8050
REMOTE_PORT=8050

echo "🚀 设置SSH隧道连接到远程开发环境..."
echo "🌐 服务器: $SERVER_IP"
echo "🔌 端口映射: 本地 $LOCAL_PORT → 服务器 $REMOTE_PORT"
echo ""

echo "🔍 检查远程服务器状态..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes root@$SERVER_IP 'docker-compose ps' 2>/dev/null | grep -q "Up"; then
    echo "✅ 远程服务正在运行"
else
    echo "⚠️  远程服务可能未运行"
    echo "💡 建议先在服务器上启动服务:"
    echo "   ssh root@$SERVER_IP"
    echo "   cd /opt/ibkr-options-platform"
    echo "   docker-compose up -d"
    echo ""
    read -p "是否继续设置隧道连接? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🔗 建立SSH隧道 (按Ctrl+C退出)..."
echo "🌐 本地访问: http://localhost:$LOCAL_PORT"
echo ""

# 设置SSH隧道
ssh -NL $LOCAL_PORT:localhost:$REMOTE_PORT root@$SERVER_IP
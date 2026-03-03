#!/bin/bash
# 代码同步脚本 - 将本地更改同步到服务器

set -e

SERVER_IP="8.148.154.206"
LOCAL_PATH="/mnt/harddisk/lwb/options-trading-platform"
REMOTE_PATH="/opt/ibkr-options-platform"

echo "🔄 同步本地代码到服务器..."

# 检查本地是否有未提交的更改
cd $LOCAL_PATH
UNCOMMITTED=$(git status --porcelain | wc -l)

if [ $UNCOMMITTED -gt 0 ]; then
    echo "📝 检测到未提交的更改:"
    git status --porcelain
    
    read -p "是否提交所有更改? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "⚠️ 未提交更改将不会同步到服务器"
    fi
fi

# 推送到远程仓库
echo "📤 推送代码到远程仓库..."
git push origin main

# 在服务器上拉取最新代码
echo "📥 在服务器上拉取最新代码..."
ssh root@$SERVER_IP "
    cd $REMOTE_PATH
    git pull origin main
    echo '✅ 代码已同步到服务器'
"

# 询问是否重启服务
echo ""
read -p "是否重启服务器上的Docker服务以应用更改? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 重启服务器上的Docker服务..."
    ssh root@$SERVER_IP "
        cd $REMOTE_PATH
        docker-compose down
        docker-compose up -d
        echo '✅ 服务已重启'
    "
    echo "🔍 检查服务状态..."
    sleep 5
    ssh root@$SERVER_IP "
        cd $REMOTE_PATH
        docker-compose ps
    "
fi

echo "✅ 代码同步完成！"
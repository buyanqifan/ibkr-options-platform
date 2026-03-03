# 本地开发环境切换指南

## 🎯 环境切换说明

### 1. 服务器环境（推荐）
**用途**: 生产环境，已配置好IBKR连接
**位置**: 8.148.154.206
**路径**: /opt/ibkr-options-platform

**访问方式**:
```bash
# SSH连接到服务器
ssh root@8.148.154.206

# 进入项目目录
cd /opt/ibkr-options-platform

# 检查服务状态
docker-compose ps

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs --follow app
```

### 2. 本地开发环境
**用途**: 代码编辑和开发
**路径**: /mnt/harddisk/lwb/options-trading-platform

**使用方式**:
```bash
# 编辑代码
# 所有代码修改都在本地进行

# 通过SSH隧道访问服务器服务
./ssh_tunnel.sh
```

## 🔧 远程开发工作流

### 1. 代码开发
- 在本地编辑代码
- 通过Git同步到服务器

### 2. 服务访问
- 服务器运行Docker服务
- 本地通过SSH隧道访问Web界面

### 3. 调试流程
```bash
# 1. 在本地修改代码
# 2. 提交到Git仓库
# 3. 在服务器上拉取最新代码
# 4. 重启Docker服务
# 5. 通过SSH隧道访问验证
```

## 📋 服务器操作命令

### 检查服务状态
```bash
ssh root@8.148.154.206 "cd /opt/ibkr-options-platform && docker-compose ps"
```

### 重启服务
```bash
ssh root@8.148.154.206 "cd /opt/ibkr-options-platform && docker-compose down && docker-compose up -d"
```

### 查看日志
```bash
ssh root@8.148.154.206 "cd /opt/ibkr-options-platform && docker-compose logs --tail=20 app"
```

### 更新代码
```bash
ssh root@8.148.154.206 "cd /opt/ibkr-options-platform && git pull && docker-compose down && docker-compose up -d"
```

## 🌐 SSH隧道访问
```bash
# 启动SSH隧道
./ssh_tunnel.sh

# 在浏览器中访问
http://localhost:8050
```

## 💡 开发技巧

1. **代码同步**: 使用Git进行代码同步
2. **即时预览**: 通过SSH隧道实时查看服务器上的应用
3. **远程调试**: 服务器上运行的服务可以连接到IBKR
4. **本地编辑**: 在本地舒适地编辑代码

## 🚀 快速开始

1. 确保服务器服务已启动
2. 运行 `./ssh_tunnel.sh` 建立隧道
3. 在浏览器中访问 `http://localhost:8050`
4. 开始开发工作

---
*这份指南帮助您在本地开发环境和远程服务器之间无缝切换*
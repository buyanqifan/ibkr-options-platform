#!/usr/bin/env python3
"""
本地IBKR连接测试脚本
用于测试本地IBKR连接是否正常
"""

import socket
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from core.ibkr.event_bridge import AsyncEventBridge
from core.ibkr.connection import IBKRConnectionManager
from utils.logger import setup_logger

logger = setup_logger("local_test")

def test_local_connection():
    """测试本地IBKR连接"""
    print("🔍 测试本地IBKR连接...")
    print(f"   Host: {settings.IBKR_HOST}")
    print(f"   Port: {settings.IBKR_PORT}")
    print(f"   Client ID: {settings.IBKR_CLIENT_ID}")
    print()
    
    # 首先测试网络连接
    print("📡 测试网络连通性...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((settings.IBKR_HOST, settings.IBKR_PORT))
        sock.close()
        
        if result == 0:
            print("✅ 网络连接正常")
        else:
            print("❌ 网络连接失败")
            print("💡 请确保:")
            print("   - TWS或IB Gateway已启动")
            print("   - API设置已启用")
            print("   - 允许来自localhost的连接")
            print("   - 端口配置正确 (4002纸币交易, 4001实盘交易)")
            return False
    except Exception as e:
        print(f"❌ 网络测试失败: {e}")
        return False
    
    # 测试IBKR连接
    print("\n🔐 测试IBKR连接...")
    try:
        bridge = AsyncEventBridge()
        bridge.start()
        conn_mgr = IBKRConnectionManager(bridge)
        
        success = conn_mgr.connect(
            host=settings.IBKR_HOST,
            port=settings.IBKR_PORT,
            client_id=settings.IBKR_CLIENT_ID
        )
        
        if success:
            print("✅ IBKR连接成功!")
            status = conn_mgr.status
            print(f"   状态: {status.state.value}")
            print(f"   账户: {status.account or 'N/A'}")
            print(f"   服务器版本: {status.server_version}")
            print(f"   消息: {status.message}")
            
            # 尝试获取账户数据
            try:
                from core.ibkr.data_client import IBKRDataClient
                from core.market_data.cache import DataCache
                
                cache = DataCache()
                data_client = IBKRDataClient(conn_mgr, bridge, cache)
                
                print("\n💰 测试获取账户数据...")
                summary = data_client.get_account_summary()
                if summary:
                    print("✅ 账户数据获取成功:")
                    for key, value in summary.items():
                        print(f"   {key}: {value}")
                else:
                    print("⚠️  无法获取账户数据，但连接正常")
                    
            except Exception as e:
                print(f"⚠️  数据获取测试失败: {e}")
            
            # 断开连接
            conn_mgr.disconnect()
            return True
        else:
            print("❌ IBKR连接失败")
            status = conn_mgr.status
            print(f"   状态: {status.state.value}")
            print(f"   错误: {status.message}")
            return False
            
    except Exception as e:
        print(f"❌ IBKR连接测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 加载本地配置
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    success = test_local_connection()
    sys.exit(0 if success else 1)
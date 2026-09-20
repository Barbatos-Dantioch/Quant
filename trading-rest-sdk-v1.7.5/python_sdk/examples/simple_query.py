#!/usr/bin/env python3
"""
简单查询示例

演示如何查询DECODED行情数据
"""

import os
import sys

# 如果SDK未安装，添加路径
sys.path.insert(0, '/opt/demo/rest/python_sdk')

from trading_rest_sdk import TradingRestClient

# 配置
API_KEY = os.environ.get('REST_API_KEY')
if not API_KEY:
    print("❌ 错误：未设置API Key")
    print("请先设置环境变量：export REST_API_KEY=your_api_key")
    exit(1)


def main():
    # 创建客户端（使用默认公网地址）
    client = TradingRestClient(api_key=API_KEY)
    
    # 如果在内网环境，需要指定内网地址：
    # client = TradingRestClient(
    #     api_key=API_KEY,
    #     base_url="http://192.168.20.10:8080"
    # )
    
    print("=" * 60)
    print("简单查询示例 - 查询DECODED行情数据")
    print("=" * 60)
    print()
    
    # 查询单条数据
    print("📝 查询平安银行 2025-09-30 09:30 的快照数据...")
    try:
        data = client.query_decoded(
            message_type="ZZ-01",
            symbol="SZ.000001",
            date="20250930",
            minute="0930"
        )
        
        print("✅ 查询成功！")
        print(f"数据类型: {type(data)}")
        print(f"数据内容: {data}")
        print()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        print()
    
    
    # 获取数据目录
    print("📝 获取数据目录...")
    try:
        catalog = client.get_catalog()
        print("✅ 数据目录获取成功！")
        print(f"版本: {catalog.version}")
        print(f"数据类型数量: {len(catalog.data_types)}")
        print()
        
    except Exception as e:
        print(f"❌ 获取目录失败: {e}")
        print()
    
    print("=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

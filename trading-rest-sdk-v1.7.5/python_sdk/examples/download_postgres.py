#!/usr/bin/env python3
"""
下载PostgreSQL表数据示例

演示如何下载静态数据表（710张表）
"""

import os
import sys

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
    
    print("=" * 70)
    print("PostgreSQL表数据下载示例")
    print("=" * 70)
    print()
    
    # 示例1：下载CSV格式
    print("📝 示例1: 下载股票业绩表（CSV格式）...")
    try:
        csv_file = client.download_postgres_table(
            table="mkt_equ_perf",
            columns=["TICKER_SYMBOL", "TRADE_DATE", "CHG_PCT", "CHG_PCT_1M"],
            date_range={
                "start_date": "2025-09-01",
                "end_date": "2025-09-30",
                "date_field": "TRADE_DATE"
            },
            limit=100,  # 限制100条（测试用）
            format="csv"
        )
        
        print(f"✅ CSV下载成功: {csv_file}")
        
        # 显示文件大小
        import os
        file_size = os.path.getsize(csv_file)
        print(f"   文件大小: {file_size / 1024:.2f} KB")
        print()
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print()
    
    # 示例2：下载JSON格式
    print("📝 示例2: 下载JSON格式数据...")
    try:
        json_data = client.download_postgres_table(
            table="mkt_equ_perf",
            columns=["TICKER_SYMBOL", "TRADE_DATE", "CHG_PCT"],
            limit=10,
            format="json"
        )
        
        print(f"✅ JSON下载成功")
        print(f"   数据类型: {type(json_data)}")
        print(f"   记录数: {len(json_data) if isinstance(json_data, list) else 'N/A'}")
        print()
        
        # 显示前2条
        if isinstance(json_data, list) and len(json_data) > 0:
            print("   前2条数据:")
            for i, record in enumerate(json_data[:2]):
                print(f"   [{i+1}] {record}")
        print()
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print()
    
    print("=" * 70)
    print("💡 提示：")
    print("1. 支持710张表的下载")
    print("2. 可以筛选字段、添加条件、设置日期范围")
    print("3. 使用pandas分析CSV，参考 pandas_tutorial.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

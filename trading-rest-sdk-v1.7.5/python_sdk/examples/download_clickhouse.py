#!/usr/bin/env python3
"""
ClickHouse数据源使用示例

演示如何使用REST SDK下载ClickHouse加工数据
"""

from trading_rest_sdk import TradingRestClient


def main():
    # 创建客户端（使用默认公网地址）
    # 内网用户请使用: base_url="http://192.168.20.10:8080"
    client = TradingRestClient(
        api_key="your-api-key-here",
        # base_url="http://192.168.20.10:8080"  # 内网地址
    )
    
    print("=" * 80)
    print("ClickHouse数据源使用示例")
    print("=" * 80)
    
    # 示例1: 下载ClickHouse表数据（CSV格式）
    print("\n1. 下载ClickHouse表数据（CSV格式）")
    print("-" * 80)
    
    try:
        csv_file = client.download_postgres_table(
            table="zz_200",  # ClickHouse表名
            columns=["trade_date", "stock_code", "stock_name", "prev_close", "market"],
            date_range={
                "start_date": "2025-10-01",
                "end_date": "2025-10-14"
            },
            datasource="clickhouse",  # 🆕 指定使用ClickHouse
            format="csv",
            progress_callback=lambda p, m: print(f"  进度: {p}% - {m}")
        )
        
        print(f"\n✅ 下载成功: {csv_file}")
        print(f"   查看前5行:")
        with open(csv_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"   {line.strip()}")
                else:
                    break
    
    except Exception as e:
        print(f"❌ 下载失败: {e}")
    
    # 示例2: 下载ClickHouse表数据（JSON格式）
    print("\n2. 下载ClickHouse表数据（JSON格式）")
    print("-" * 80)
    
    try:
        data = client.download_postgres_table(
            table="zz_200",
            columns=["stock_code", "stock_name", "prev_close"],
            datasource="clickhouse",  # 使用ClickHouse
            format="json"
        )
        
        print(f"✅ 下载成功，共 {len(data.get('data', []))} 条记录")
        print(f"   前3条数据:")
        for i, record in enumerate(data.get('data', [])[:3]):
            print(f"   {i+1}. {record}")
    
    except Exception as e:
        print(f"❌ 下载失败: {e}")
    
    # 示例3: 带条件筛选的下载
    print("\n3. 带条件筛选的ClickHouse数据下载")
    print("-" * 80)
    
    try:
        csv_file = client.download_postgres_table(
            table="zz_200",
            conditions={
                "market": "深圳主板"  # 只下载深圳主板的股票
            },
            date_range={
                "start_date": "2025-10-10",
                "end_date": "2025-10-14"
            },
            order_by="prev_close DESC",  # 按昨收价倒序
            datasource="clickhouse",
            format="csv"
        )
        
        print(f"✅ 筛选下载成功: {csv_file}")
    
    except Exception as e:
        print(f"❌ 下载失败: {e}")
    
    # 示例4: 对比PostgreSQL和ClickHouse
    print("\n4. 数据源对比")
    print("-" * 80)
    
    print("PostgreSQL:")
    print("  - 755张原始静态数据表")
    print("  - 财务数据、股票信息、经济数据等")
    print("  - 适合：完整的原始数据查询")
    
    print("\nClickHouse:")
    print("  - 加工后的宽表和汇总数据")
    print("  - 高性能OLAP查询")
    print("  - 适合：大数据量分析、多表关联")
    
    print("\n使用建议:")
    print("  - 下载原始表数据 → 使用PostgreSQL")
    print("  - 下载加工宽表 → 使用ClickHouse (datasource='clickhouse')")


if __name__ == "__main__":
    main()


"""
ClickHouse行情镜像库下载示例
演示如何从ClickHouse行情镜像库下载数据
"""

from trading_rest_sdk import TradingRestClient

# 初始化客户端
client = TradingRestClient(
    api_key="your_api_key_here",
    base_url="http://61.151.241.233:8080"
)

print("========================================")
print("ClickHouse行情镜像库数据下载")
print("========================================\n")

# 1. 查看行情镜像库的表
print("【1】查看行情镜像库的表列表")
tables = client.list_tables(datasource="clickhouse_data")
print(f"   表数量: {len(tables)}\n")

# 显示几个示例表
print("   示例表:")
for i, table in enumerate(tables[:5], 1):
    print(f"   {i}. {table['table_name']:15s} - {table['table_comment']}")

# 2. 下载深圳股票快照数据（zz_01）
print("\n【2】下载深圳股票快照历史数据")
csv_file = client.download_postgres_csv(
    table="zz_01",
    columns=[],  # 所有字段
    date_range={
        "start_date": "2024-11-01",
        "end_date": "2024-11-15"
    },
    datasource="clickhouse_data"  # 指定行情镜像库
)
print(f"   ✅ 下载完成: {csv_file}")

# 3. 使用SQL查询
print("\n【3】使用SQL查询指数K线数据")
result = client.execute_sql(
    sql="""
        SELECT * FROM zz_7001 
        WHERE zz_date = '20251115' 
        AND index_code|指数代码 = 'SH.000001'
        LIMIT 10
    """,
    datasource="clickhouse_data"
)
print(f"   查询结果: {result['count']}条")
if result['data']:
    print(f"   字段数: {len(result['data'][0])}个")

# 4. 按市场分类查询
print("\n【4】查看行情镜像库的市场分类")
response = client._request("GET", "/api/v1/dbdict/categories", params={"datasource": "clickhouse_data"})
categories = response.json().get('data', [])
print(f"   分类数: {len(categories)}\n")
for cat in categories:
    print(f"   - {cat['name']}: {cat['table_count']}张表")

print("\n✅ 行情镜像库使用示例完成！")


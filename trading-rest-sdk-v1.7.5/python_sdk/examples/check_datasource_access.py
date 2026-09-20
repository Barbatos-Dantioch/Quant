"""
数据源权限查询示例
演示如何查询当前用户可访问的数据源
"""

from trading_rest_sdk import TradingRestClient

# 初始化客户端
client = TradingRestClient(
    api_key="your_api_key_here",
    base_url="http://61.151.241.233:8080"
)

print("========================================")
print("查询用户可访问的数据源")
print("========================================\n")

# 获取数据源列表（根据API Key权限）
datasources = client.get_datasources()

print(f"系统共有 {len(datasources)} 个数据源\n")

# 统计可用和不可用的数据源
available_count = sum(1 for ds in datasources if ds['available'])
print(f"  ✅ 可访问: {available_count}个")
print(f"  ❌ 无权限: {len(datasources) - available_count}个\n")

# 详细列表
print("数据源详情：\n")
for ds in datasources:
    if ds['available']:
        print(f"✅ {ds['name']}")
        print(f"   代码: {ds['code']}")
        print(f"   类型: {ds['type']}")
        print(f"   表数量: {ds['tables']}张")
        print(f"   数据库: {ds['database']}")
    else:
        print(f"❌ {ds['name']}")
        print(f"   无权限访问")
    print()

print("="*40)
print("根据不同数据源查询数据：")
print("="*40)

# 根据数据源查询表
for ds in datasources:
    if ds['available'] and ds['code'] != 'redis':
        print(f"\n【{ds['name']}】")
        try:
            tables = client.list_tables(datasource=ds['code'])
            print(f"  表数量: {len(tables)}")
            if tables:
                print(f"  示例表: {tables[0]['table_name']}")
        except Exception as e:
            print(f"  查询失败: {e}")


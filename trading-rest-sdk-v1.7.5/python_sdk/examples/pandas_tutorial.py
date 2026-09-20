#!/usr/bin/env python3
"""
Pandas数据分析教程

演示如何使用pandas分析下载的数据
注意：需要先安装pandas: pip install pandas
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
    print("=" * 70)
    print("📊 Pandas数据分析教程")
    print("=" * 70)
    print()
    
    # 检查pandas是否安装
    try:
        import pandas as pd
        print(f"✅ pandas版本: {pd.__version__}")
        print()
    except ImportError:
        print("❌ 错误：pandas未安装")
        print("请先安装: pip install pandas")
        print()
        return
    
    # 创建客户端（使用默认公网地址）
    client = TradingRestClient(api_key=API_KEY)
    
    # 如果在内网环境，需要指定内网地址：
    # client = TradingRestClient(
    #     api_key=API_KEY,
    #     base_url="http://192.168.20.10:8080"
    # )
    
    # ========== 步骤1: 下载CSV文件 ==========
    print("📝 步骤1: 下载PostgreSQL表数据...")
    try:
        csv_file = client.download_postgres_table(
            table="mkt_equ_perf",
            columns=["TICKER_SYMBOL", "TRADE_DATE", "CHG_PCT", "CHG_PCT_1M"],
            limit=100,  # 测试用，只下载100条
            format="csv"
        )
        print(f"✅ 下载成功: {csv_file}")
        print()
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return
    
    # ========== 步骤2: 使用pandas读取 ==========
    print("📝 步骤2: 使用pandas读取CSV...")
    df = pd.read_csv(csv_file)
    print(f"✅ 读取成功，DataFrame形状: {df.shape}")
    print(f"   ({df.shape[0]}行 x {df.shape[1]}列)")
    print()
    
    # ========== 步骤3: 查看数据 ==========
    print("📝 步骤3: 查看数据前5行...")
    print(df.head())
    print()
    
    # ========== 步骤4: 基础统计 ==========
    print("📝 步骤4: 基础统计信息...")
    print(df.describe())
    print()
    
    # ========== 步骤5: 数据筛选 ==========
    print("📝 步骤5: 筛选涨幅大于5%的股票...")
    rising_stocks = df[df['CHG_PCT'] > 5.0]
    print(f"找到 {len(rising_stocks)} 只涨幅大于5%的股票")
    if len(rising_stocks) > 0:
        print(rising_stocks[['TICKER_SYMBOL', 'TRADE_DATE', 'CHG_PCT']].head())
    print()
    
    # ========== 步骤6: 数据计算 ==========
    print("📝 步骤6: 计算平均涨跌幅...")
    avg_chg = df['CHG_PCT'].mean()
    print(f"平均涨跌幅: {avg_chg:.2f}%")
    print()
    
    # ========== 步骤7: 排序 ==========
    print("📝 步骤7: 按涨跌幅排序（前5名）...")
    top_5 = df.nlargest(5, 'CHG_PCT')
    print(top_5[['TICKER_SYMBOL', 'TRADE_DATE', 'CHG_PCT']])
    print()
    
    # ========== 步骤8: 保存处理后的数据 ==========
    print("📝 步骤8: 保存处理后的数据...")
    output_file = "downloads/processed_data.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ 已保存到: {output_file}")
    print()
    
    print("=" * 70)
    print("🎉 教程完成！")
    print("=" * 70)
    print()
    print("💡 Pandas常用操作：")
    print("   - df.head() - 查看前几行")
    print("   - df[df['列'] > 值] - 筛选数据")
    print("   - df['列'].mean() - 计算平均值")
    print("   - df.groupby('列').mean() - 分组统计")
    print("   - df.sort_values('列') - 排序")
    print("   - df.to_csv() - 保存为CSV")
    print("   - df.to_excel() - 保存为Excel")
    print()
    print("📚 更多pandas教程：https://pandas.pydata.org/docs/")
    print("=" * 70)


if __name__ == "__main__":
    main()

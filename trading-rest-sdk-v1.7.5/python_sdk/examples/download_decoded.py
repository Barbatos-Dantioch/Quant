#!/usr/bin/env python3
"""
下载DECODED数据示例

演示如何下载历史行情数据为CSV文件
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
    client = TradingRestClient(
        api_key=API_KEY,
        download_dir="./downloads"  # 下载目录
    )
    
    # 如果在内网环境，需要指定内网地址：
    # client = TradingRestClient(
    #     api_key=API_KEY,
    #     base_url="http://192.168.20.10:8080",
    #     download_dir="./downloads"
    # )
    
    print("=" * 70)
    print("DECODED数据下载示例")
    print("=" * 70)
    print()
    
    print("📝 下载平安银行9月份的股票快照数据...")
    print("   消息类型: ZZ-01 (深圳股票快照)")
    print("   股票: SZ.000001 (平安银行)")
    print("   日期范围: 2025-09-01 至 2025-09-30")
    print()
    
    try:
        # 一站式下载（自动创建任务、等待、下载）
        csv_file = client.download_decoded_csv(
            message_type="ZZ-01",
            symbols=["SZ.000001"],
            date_range={"start": "20250901", "end": "20250930"}
        )
        
        print(f"✅ 下载成功！")
        print(f"文件路径: {csv_file}")
        print()
        
        # 显示文件信息
        import os
        file_size = os.path.getsize(csv_file)
        print(f"文件大小: {file_size / 1024:.2f} KB")
        
        # 显示前几行
        print("\n📄 文件内容预览（前5行）:")
        print("-" * 70)
        with open(csv_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(line.rstrip())
        print("-" * 70)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 70)
    print("提示：可以使用pandas读取CSV文件进行数据分析")
    print("参考 examples/pandas_tutorial.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

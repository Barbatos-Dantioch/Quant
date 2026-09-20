"""
Trading REST SDK 安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="trading-rest-sdk",
    version="1.7.5",
    author="Trading Team",
    author_email="support@trading.com",
    description="Python SDK for Trading REST API - 金融市场数据查询和下载客户端",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trading/rest-sdk-python",
    packages=find_packages(exclude=["tests", "examples", "tools"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "pandas": ["pandas>=1.5.0"],  # 可选：DataFrame支持
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    keywords=[
        "rest api",
        "trading",
        "stock",
        "market data",
        "financial",
        "query",
        "download",
        "行情",
        "股票",
        "数据下载",
    ],
    project_urls={
        "Documentation": "https://github.com/trading/rest-sdk-python/docs",
        "Source": "https://github.com/trading/rest-sdk-python",
    },
)

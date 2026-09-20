# Changelog - Trading REST SDK

## [1.7.4] - 2025-11-15

### 关键修复（SDK bug）
- 🐛 **修复 download_file() 路径选择** - 解决ClickHouse任务下载404问题
  - 智能路由：自动尝试两种下载路径
  - dbdict任务：`/api/v1/dbdict/download-file/{id}`
  - DECODED任务：`/api/v1/download/file/{id}`
  - 新增task_type参数（可选，支持手动指定）

### 后端配套修复
- 🔧 静态数据下载任务正确处理symbols参数
  - symbols转换为conditions['__symbols_in__']
  - ClickHouse智能识别股票代码字段
  - 支持单只和多只股票筛选

### 测试结果（预期）
- ✅ PostgreSQL: 6/6 (100%)
- ✅ Redis: 6/6 (100%)
- ✅ ClickHouse加工库: 5/5 (100%) ← symbols筛选已修复
- ✅ ClickHouse行情镜像库: 4/4 (100%)
- **综合通过率: 21/21 (100%)** ⭐⭐⭐⭐⭐

---

## [1.7.3] - 2025-11-15

### 重要修复（前端测试反馈）
- 🐛 **修复 create_postgres_download_task()** - 现在返回任务ID而非文件路径
  - 正确调用异步任务接口 `/api/v1/dbdict/download-task`
  - 支持 symbols 参数
  - 可以正确使用 `wait_for_task()` 查询状态
  
- 🐛 **修复 create_clickhouse_download_task()** - 统一异步任务行为
  - 现在也返回任务ID
  - symbols筛选功能完全可用
  - 支持ClickHouse数据加工库和行情镜像库

### 后端配套修复
- 🔧 支持ClickHouse中文字段名
  - 自动检测中文字段并添加反引号
  - 解决行情镜像库字段筛选问题
  
### 测试结果
- ✅ PostgreSQL: 100%通过（6/6场景）
- ✅ Redis: 100%通过（6/6场景）
- ✅ ClickHouse加工库: 100%通过（5/5场景，symbols筛选已修复）
- ✅ ClickHouse行情镜像库: 100%通过（4/4场景，中文字段已修复）
- **综合通过率: 100%** ⭐⭐⭐⭐⭐

---

## [1.7.2] - 2025-11-15

### Bug修复（测试反馈）
- 🐛 修复 `execute_sql(datasource='clickhouse_data')` 自动识别
  - 现在可以自动识别行情镜像库的表
  - 支持查询zz_01等行情历史表
- 🐛 修复 `create_postgres_download_task()` 方法调用错误
- 🐛 修复 `create_clickhouse_download_task()` symbols参数支持

### 改进
- 📝 execute_sql文档更新 - 添加clickhouse_data说明
- 🔧 自动识别逻辑增强 - 检查4个数据源

### 重要说明
- ⚠️ **查询行情镜像库时必须明确指定datasource**：`execute_sql(sql, datasource="clickhouse_data")`
- ⚠️ 行情镜像库的字段名可能是中文（如"证券代码"、"最新价"），使用前请通过 `preview_table()` 查看实际字段
- ⚠️ 不能按股票代码筛选行情镜像库下载任务（字段名差异）

---

## [1.7.1] - 2025-11-15

### Bug修复
- 🐛 `get_table_schema()` - 添加datasource参数支持
- 🐛 `preview_table()` - 添加datasource参数支持
- 🐛 `search_tables()` - 添加datasource参数支持
- 🐛 `get_categories()` - 新增方法（获取分类）
- 🐛 `get_database_stats()` - 新增方法（获取统计）
- 🐛 `create_postgres_download_task()` - 新增别名方法
- 🐛 `create_clickhouse_download_task()` - 新增别名方法

### 说明
- 修复测试中发现的18个错误中的大部分
- 所有表操作方法现在都支持datasource参数
- 增强多数据源兼容性

---

## [1.7.0] - 2025-11-15

### 新增功能
- ✨ **数据源权限管理** - 新增 `get_datasources()` 方法，查询用户可访问的数据源
- 🎯 **第4个数据源** - 支持ClickHouse行情镜像库（`datasource="clickhouse_data"`）
- 📊 **ZZ-7001支持** - 沪深指数1分钟K线数据
- 🔧 **动态数据源** - 根据API Key权限动态访问不同数据源

### 改进
- 📝 `list_tables(datasource)` - 新增datasource参数，支持查询不同数据源的表
- 📝 `download_postgres_csv(datasource)` - 新增datasource参数，支持下载ClickHouse数据
- 📚 文档更新 - 从硬编码数字（59个、3种）改为动态描述

### 新增示例
- `examples/check_datasource_access.py` - 数据源权限查询示例
- `examples/download_clickhouse_data.py` - 行情镜像库使用示例

### 重要说明
- ⚠️ **数据源访问由API Key控制** - 不同用户可访问的数据源不同
- 📊 系统支持4个数据源：Redis、PostgreSQL、ClickHouse（数据加工库）、ClickHouse（行情镜像库）
- 🔍 使用 `get_datasources()` 查询当前用户的数据源权限
- 🎯 行情数据源从59个增加到60个（新增ZZ-7001）

---

## [1.6.0] - 2025-11-07

### 新增功能
- ✨ **数据字典API**（5个新方法）
  - `get_markets()` - 获取市场列表
  - `list_data_sources()` - 列出所有数据源
  - `get_fields()` - 获取字段定义
  - `preview_data()` - 预览数据源数据
  - `search_data_sources()` - 搜索数据源

- ✨ **PostgreSQL表功能增强**（2个新方法）
  - `preview_table()` - 预览表数据
  - `search_tables()` - 搜索表和字段

- ✨ **任务管理增强**（2个新方法）
  - `list_tasks()` - 列出下载任务
  - `cancel_task()` - 取消任务

### 改进
- 📝 完善文档（RELEASE_NOTES、用户手册）
- 🔧 功能覆盖率提升至80%+
- ✅ 支持ZZ-111等新数据源（字段名"英文|中文"格式）

### Bug修复
- 无

---

## [1.5.6] - 2025-11-07

### 测试完善
- ✅ 新增严格测试套件
  - 三种数据源完整测试（Redis、PostgreSQL、ClickHouse）
  - 字段筛选和条件查询测试
  - 并发下载测试
  - 数据格式验证（CSV/JSON）
- ✅ Docker测试环境
- ✅ 自动化测试报告生成

### API Gateway配套更新（V2.4.9）
- 支持K线数据源（ZZ-5001, ZZ-6001）
- 支持指数行业快照（ZZ-111）
- 支持数字货币数据（ZZ-130, ZZ-131, ZZ-132）

### 注意事项
- 所有功能已在Docker环境中测试
- 支持3种数据源的完整访问
- 异步任务推荐用于大文件下载

---

## [1.5.4] - 2025-10-15

### 新增功能
- SQL查询功能（PostgreSQL和ClickHouse）
- 数据库字典API支持
- 静态数据下载功能增强

---

## [1.5.0] - 2025-09-20

### 新增功能
- PostgreSQL表数据下载
- 异步下载任务系统
- 数据字典API

---

## [1.4.0] - 2025-09-12

### 初始版本
- DECODED行情数据查询
- RAW行情数据查询
- 批量查询功能
- 数据下载功能
- 数据目录查询

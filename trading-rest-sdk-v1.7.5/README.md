# SDK下载中心

本目录存放所有可供客户下载的SDK开发包。

## 📦 当前可用SDK

| SDK类型 | 版本 | 文件名 | 大小 | 更新日期 |
|---------|------|--------|------|----------|
| Python REST SDK | v1.4.0 | trading-rest-sdk-v1.4.0.tar.gz | 16KB | 2025-10-09 |
| Python WebSocket SDK | v1.0.0 | trading-websocket-sdk-v1.0.0.tar.gz | 47KB | 2025-09-30 |

## 🔌 API接口

### 获取SDK列表（公开访问，无需认证）
```bash
GET http://61.151.241.233:8080/api/v1/sdk/versions
```

**返回示例：**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "sdks": [
      {
        "type": "python-rest",
        "name": "Python REST SDK",
        "version": "1.4.0",
        "filename": "trading-rest-sdk-v1.4.0.tar.gz",
        "size": 16124,
        "md5": "dd06b1542607ee7ae06a0c294a6177d5",
        "download_url": "http://61.151.241.233:8080/downloads/sdk/trading-rest-sdk-v1.4.0.tar.gz"
      }
    ]
  }
}
```

### 下载SDK文件（公开访问，无需认证）
```bash
GET http://61.151.241.233:8080/downloads/sdk/{filename}
```

**示例：**
```bash
# 下载REST SDK
curl -O http://61.151.241.233:8080/downloads/sdk/trading-rest-sdk-v1.4.0.tar.gz

# 下载WebSocket SDK
curl -O http://61.151.241.233:8080/downloads/sdk/trading-websocket-sdk-v1.0.0.tar.gz
```

## 📝 更新SDK流程

**运维人员更新SDK的步骤：**

### 1. 上传新版SDK文件
```bash
# 将新版SDK文件放到本目录
cp /path/to/trading-rest-sdk-v1.5.0.tar.gz /opt/api_gateway/downloads/sdk/
```

### 2. 更新版本配置
编辑 `sdk-versions.json`：
```json
{
  "sdks": [
    {
      "type": "python-rest",
      "name": "Python REST SDK",
      "version": "1.5.0",                              // 更新版本号
      "release_date": "2025-10-10T10:00:00Z",          // 更新日期
      "description": "...",
      "filename": "trading-rest-sdk-v1.5.0.tar.gz"     // 更新文件名
    }
  ],
  "update_time": "2025-10-10T10:00:00Z"                // 更新时间
}
```

### 3. 保存配置

**无需重启API服务，配置立即生效！**

客户端下次查询会自动获取最新版本信息。

## 🔒 安全特性

- ✅ **路径安全**：防止 `../` 路径遍历攻击
- ✅ **文件校验**：提供MD5值供客户端验证
- ✅ **访问控制**：公开访问（不需要API Key）
- ✅ **错误处理**：404友好提示

## 📊 技术实现

**后端读取流程：**
1. 读取 `sdk-versions.json` 配置
2. 遍历每个SDK，读取文件信息（大小、MD5）
3. 自动拼接下载URL
4. 返回完整的SDK列表

**下载流程：**
1. 验证文件名（防止路径遍历）
2. 检查文件存在性
3. 设置响应头（Content-Disposition）
4. 返回文件流

## 🛠️ 维护命令

```bash
# 查看目录内容
ls -lh /opt/api_gateway/downloads/sdk/

# 查看配置文件
cat /opt/api_gateway/downloads/sdk/sdk-versions.json

# 测试API
curl -s http://192.168.20.10:8080/api/v1/sdk/versions | python3 -m json.tool

# 测试下载
curl -I http://192.168.20.10:8080/downloads/sdk/trading-rest-sdk-v1.4.0.tar.gz
```

## 📞 联系方式

如有问题，请联系技术支持团队。

---

**目录位置**：`/opt/api_gateway/downloads/sdk/`  
**配置文件**：`sdk-versions.json`  
**更新时间**：2025-10-09  
**维护人员**：运维团队

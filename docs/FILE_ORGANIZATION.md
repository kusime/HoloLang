# 项目文件组织说明

## 目录结构

```
HoloLang/
├── app/                    # 应用核心代码
├── tests/                  # 测试文件
├── docs/                   # 文档和归档
│   ├── archive/           # 旧文件归档
│   └── *.md, *.yml        # 依赖配置文档
├── scripts/               # 工具脚本
│   └── examples/         # 示例脚本
├── backups/              # 备份文件
└── minio/                # MinIO 数据
```

## 文件移动记录

### 移动到 `docs/archive/`
- `basicinfo_refactor.md` - 重构笔记（已过时）
- `text.txt` - 测试文本文件
- `preview.html` - 预览测试文件
- `ARCHITECTURE.md` - 架构文档（已合并到 README）
- `WHISPERX_SETUP.md` - WhisperX 配置文档（已合并到 README）

### 移动到 `docs/`
- `holo-align-*.txt` - 依赖列表文件
- `holo-align-*.yml` - Conda 环境配置

### 保留在根目录
- `README.md` - **统一项目文档**（合并了原 ARCHITECTURE.md 和 WHISPERX_SETUP.md）
- `requirements.txt` - Python 依赖
- `docker-compose.yml` - Docker 配置
- `.env`, `.gitignore`, `activate.sh` - 配置文件

## 清理后的根目录

根目录现在只保留：
- ✅ **单一文档**（README.md）
- ✅ 配置文件（requirements.txt, docker-compose.yml）
- ✅ 环境配置（.env, activate.sh）
- ✅ 核心目录（app/, tests/, docs/, scripts/, backups/, minio/）

# fastText 模型下载说明

## 模型文件

本项目使用 **fastText lid.176.bin** 语言识别模型。

### 自动下载（推荐）

首次运行时，模型会自动下载：

```bash
./activate.sh
```

### 手动下载

如果自动下载失败，可以手动下载：

```bash
# 下载模型（126MB）
wget -O models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# 或使用压缩版（<1MB，稍逊一筹）
wget -O models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

### 模型信息

- **文件大小**: 126MB (lid.176.bin) 或 <1MB (lid.176.ftz)
- **支持语言**: 176 种
- **准确度**: 中日英 98-100%
- **推理速度**: <1ms
- **GPU**: 不需要

## 许可证

fastText 模型由 Facebook AI Research 提供，遵循 [Creative Commons Attribution-Share-Alike License 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。

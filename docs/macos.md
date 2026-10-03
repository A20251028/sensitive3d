# macOS（Apple Silicon）构建与运行

> 适用环境：macOS arm64、Homebrew OpenSceneGraph 3.6.5、Python 3.12。
>
> 说明：以下步骤是按 macOS 写的。构建脚本、`sensitive3d doctor`、bridge 自检和超时/取消逻辑都已在 Linux（Ubuntu，OpenSceneGraph 3.6.5，Python 3.11）上实际运行；**macOS 上没有实际执行过这些命令**。所以在你的 Mac 上请以 `sensitive3d doctor` 的输出为准，并把 `sensitive3d doctor --json` 的结果附在问题反馈里。

## 1. 安装系统依赖

```bash
brew install open-scene-graph cmake python@3.12
```

## 2. 创建 Python 环境

```bash
cd /path/to/sensitive3d            # 仓库目录
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'            # zsh 下必须给 .[dev] 加引号
```

确认 Python 是 arm64（不是 Rosetta 下的 x86_64）：

```bash
python -c "import platform; print(platform.machine())"   # 应输出 arm64
```

## 3. 编译 OSGB bridge

```bash
./scripts/build_bridge.sh
```

- 脚本检测到 Homebrew（`command -v brew`）时，会自动给 CMake 加上 `-DCMAKE_PREFIX_PATH="$(brew --prefix)"`（Apple Silicon 上通常是 `/opt/homebrew`）。
- 找不到 OpenSceneGraph 时，脚本会直接退出并打印安装命令（`brew install open-scene-graph cmake`）。
- 编译完成后得到 `native/osgb_bridge/build/osgb_bridge`，脚本会接着运行一次 `osgb_bridge version`，打印 OSG 版本、插件搜索路径，以及 `.osgb`、JPEG、PNG 插件实际解析到的文件。
- 自定义 OSG 安装位置：`CMAKE_PREFIX_PATH=/your/prefix ./scripts/build_bridge.sh`。
- 如需指定使用某个 bridge 可执行文件：`export S3D_OSGB_BRIDGE=/完整路径/osgb_bridge`。设置后程序只用这一个路径，文件不存在时直接报错，不会退回到默认路径。

## 4. 环境检查（必须做）

```bash
sensitive3d doctor          # 中文可读报告
sensitive3d doctor --json   # JSON，便于存档或附在问题反馈里
```

`doctor` 会**实际执行** bridge，而不只是检查文件是否存在：

1. 导入 numpy、numba、scipy、cv2、PIL、trimesh、skimage、fast_simplification、mapbox_earcut、fastapi、uvicorn、onnxruntime（可选），列出各自版本；
2. 运行 `osgb_bridge version`：给出 OSG 版本、插件搜索路径，以及 `osgb`、`serializers_osg`、`jpg`、`png` 四个插件解析到的文件；
3. 运行 `osgb_bridge selftest`：用 OSG 插件写出并读回 JPEG 和 PNG，再生成一个带 PagedLOD 的小 `.osgb`（内嵌贴图），读回并导出，然后逐项比对像素、顶点、UV、三角形、LOD 关系和 info 扫描结果；
4. 列出 `S3D_OSGB_BRIDGE`、`OSG_LIBRARY_PATH`、`DYLD_LIBRARY_PATH`、`LD_LIBRARY_PATH`、`S3D_BRIDGE_TIMEOUT`。

退出码为 0 表示环境正常；非 0 时报告末尾会列出“问题”和“建议”。网页的健康检查只能说明可执行文件存在，以 `doctor` 的结果为准。

macOS 版 OpenSceneGraph 通常默认用 `imageio` 插件（`osgdb_imageio.so`，系统 ImageIO）读写 jpg/png，这时 doctor 里 `jpg`、`png` 两项会指向 `osgdb_imageio.so`，这是正常的；若 Homebrew 的构建带有 `osgdb_jpeg.so` / `osgdb_png.so`，则显示这两个文件。ImageIO 解出的贴图可能是 RGBA（本机先前的读取证据就是 RGBA）。对于不透明的原 JPEG 贴图，sensitive3d 修复后仍按 JPEG 写回，不会改成 PNG。自检中 PNG 像素允许最多 12 级误差（ImageIO 可能做色彩管理），JPEG 允许最多 48 级误差；四个色块的颜色差别很大，所以上下翻转、通道顺序错误或解码失败仍能被发现。

## 5. 启动网页

```bash
sensitive3d serve --host 127.0.0.1 --port 8000 --workspace workspace_macos
```

浏览器打开 <http://127.0.0.1:8000>。`--workspace` 是本次运行的独立工作目录，上传副本和输出都写在这里，不会写入原始数据目录。

建议在处理前后对原始数据做一次哈希校验，确认它没有被改动：

```bash
cd /Users/imac/sensitive3d/data
find 01Mesh 02Mesh -type f -print0 | sort -z | xargs -0 shasum -a 256 > ~/s3d_input_before.sha256
# ……处理完成后……
find 01Mesh 02Mesh -type f -print0 | sort -z | xargs -0 shasum -a 256 | diff ~/s3d_input_before.sha256 - && echo 原始数据未改变
```

## 6. 只读结构扫描（可直接复制）

`info` 只读取文件、不解码贴图，也不写任何东西。即使某个文件读不出来，它也会输出一条 `"ok": false` 记录，不会漏掉：

```bash
find /Users/imac/sensitive3d/data/01Mesh -name '*.osgb' ! -name '._*' ! -path '*/__MACOSX/*' > /tmp/s3d_list.txt
./native/osgb_bridge/build/osgb_bridge info /tmp/s3d_list.txt > /tmp/s3d_info.jsonl
python - <<'EOF'
import json
recs = [json.loads(l) for l in open("/tmp/s3d_info.jsonl")]
bad = [r for r in recs if not r["ok"]]
miss = [r["file"] for r in recs if any(g.get("texture_missing") for g in r["geometries"])]
print(len(recs), "个文件；无法读取", len(bad), "个；缺贴图", len(miss), "个")
for r in bad[:20]:
    print("  读取失败:", r["file"], r["error"])
for f in miss[:20]:
    print("  缺贴图:", f)
EOF
```

也可以在 Python 里调用 `sensitive3d.io.osgb.scan_osgb(paths)`：bridge 崩溃或记录数对不上时，它会把剩下的文件逐个重扫，保证每个路径都有一条记录。

每条记录的字段（osgb_bridge 第 2 版）：

- `geometries[]`：`has_finer`、`num_triangles`、`depth`、`min`/`max`（几何没有顶点时省略这两项）、`has_uv`、`image`（`images` 中的下标，-1 表示没有贴图）、`texture_missing`。
- `images[]`：`name`；`encoding`（`jpg`、`png`、`dds` 等，或 `raw`、`compressed`、`external`、`missing`）；`format`；`external`；`width`/`height`（直接读 JPEG SOF / PNG IHDR 头，不解码，未知时为 -1）；`bytes`；`ok`；失败时还有 `error`。
- `children`：PagedLOD 子文件名，保持文件中的原始写法。

## 7. 常见问题

### 7.1 插件找不到

现象：`doctor` 自检失败，报 `no reader for .osgb`、`Could not find plugin`、`jpeg_write` 失败；或者插件那一栏显示“未找到”。

1. 运行 `sensitive3d doctor`，看“插件搜索路径”和“插件”两栏。
2. 确认插件目录存在，而且目录名里的版本号与 doctor 显示的 OSG 版本一致：
   ```bash
   ls "$(brew --prefix)/lib/osgPlugins-3.6.5" | grep -E 'osgdb_(osg|serializers_osg|imageio|jpeg|png)\.so'
   ```
3. 把**包含** `osgPlugins-3.6.5` 的目录加入搜索路径，然后重新检查：
   ```bash
   export OSG_LIBRARY_PATH="$(brew --prefix)/lib"
   sensitive3d doctor
   ```
   确认有效后，把这一行写进 `~/.zshrc`。
4. `brew upgrade open-scene-graph` 以后插件目录的版本号可能会变，需要重新编译：`rm -rf native/osgb_bridge/build && ./scripts/build_bridge.sh`。

### 7.2 dyld 错误

现象：`dyld[...]: Library not loaded: .../libosgDB.161.dylib`、`image not found`，或 bridge 被 SIGABRT 终止。sensitive3d 的错误信息里会附上 stderr 的最后几行和处理建议。

- 原因一般是编译 bridge 之后 OpenSceneGraph 被升级、重装或卸载了。
- 处理：
  ```bash
  brew reinstall open-scene-graph
  rm -rf native/osgb_bridge/build && ./scripts/build_bridge.sh
  otool -L native/osgb_bridge/build/osgb_bridge     # 查看链接到的 libosg*.dylib 路径
  ```
- 临时绕过：`export DYLD_LIBRARY_PATH="$(brew --prefix)/lib"`。受 SIP 保护的程序（如 `/bin/sh`、`/usr/bin/env`）启动时会清掉 `DYLD_*` 变量，所以更推荐重新编译。
- `Bad CPU type in executable`：bridge 是在 Rosetta（x86_64）终端里编译的。用 `file native/osgb_bridge/build/osgb_bridge` 确认架构，应为 `arm64`；然后在原生终端里重新编译。

### 7.3 Gatekeeper

本机用 `build_bridge.sh` 编译出的可执行文件不带隔离属性（quarantine），Gatekeeper 不会拦截，不需要签名或公证。只有从别的机器或网络复制来的预编译 bridge 才可能被拦截（`xattr -l` 能看到 `com.apple.quarantine`）。这种情况建议直接在本机重新编译，不要绕过系统检查。

### 7.4 字体

合成示例和数字模板会用 TrueType 字体绘制文字。macOS 上没有 Linux 常见的字体路径（如 DejaVu），程序会退回到别的字体甚至 PIL 默认字体，导致字号和字形不同，检测/分类结果也会跟着变（例如之前本机首跑只检出 3/5）。

- 可用字体通常在 `/System/Library/Fonts`、`/System/Library/Fonts/Supplemental`、`/Library/Fonts`、`~/Library/Fonts`。
- 更换字体或升级代码后，要重新生成合成示例（`sensitive3d synth <新目录>`），不要复用旧缓存。
- 对比不同机器的结果时，记录实际使用的字体文件。

### 7.5 超时与取消

- 每次 bridge 调用默认最长 600 秒，可以用 `export S3D_BRIDGE_TIMEOUT=1800` 调整，设为 `0` 表示不限时。
- 超时或取消时，bridge 所在的整个进程组都会被结束，并抛出 `BridgeTimeout` / `BridgeCancelled`；错误信息里包含完整命令、返回码和 stderr 的最后几行。

### 7.6 坏文件与缺贴图

- 读不出的 `.osgb`（损坏、截断、不存在）在扫描结果里是 `"ok": false` 加 `error`，同一批次里的其他文件不受影响。
- 外部贴图文件缺失，或内嵌贴图无法解码：`info` 中该几何的 `texture_missing` 为 `true`，对应图片记录为 `ok: false`；`export` 不会写出损坏的 `.raw`，而是给出 `"file": null, "valid": false, "error": ...`；`read_osgb` 抛出 `BridgeTextureError`，错误信息里带文件名和贴图名。这类文件不能直接自动修复，需要人工确认。

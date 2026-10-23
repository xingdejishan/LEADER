# 快速版沿轨搜索（coarse-fine-search 分支）接入记录

2026-10-20。目的：用 GitHub 分支 `xingdejishan/LEADER@coarse-fine-search` 的粗筛+细筛实现，替换 r4 的全网格搜索。

## 1. 做了什么

- 把分支的 `src/`（`fast_lib.py`、`pipeline.py`、`run_frames.py`）拷到本目录，**代码逻辑未改**；只修了一处 Python 3.8 不支持的语法（`dict | dict` 合并，环境是 3.8，原写法会抛 `TypeError`）。
- 输入按该分支 CLI 的格式准备：`inputs/frames.json`（1502 困难帧的键与扫描路径）、`inputs/refs.json`（4752 张参考库）、`inputs/initial.npz`（初值 = 部署基线 refined_V7）。
- 输出 `results/fast.npz`（1502 帧修正位姿）与 `results/fast_diag.jsonl`（逐帧 s、d、代价）。
- 再对这批位姿跑一次 r-gate 的几何精修，得到 `results/fast_refined.npz`。

复现：

```bash
cd /home/zhangcy/projects/e2e-fast-search-20261020-v1/src
python run_frames.py --frames ../inputs/frames.json --refs ../inputs/refs.json \
    --poses ../inputs/initial.npz --out ../results/fast.npz \
    --diag ../results/fast_diag.jsonl --workers 12
python refine_fast.py ../results/fast.npz ../results/fast_refined.npz
```

## 2. 结果（1502 困难帧，统一评分器）

| 版本 | MPE | MOE | 进一米两度 | >10 米 | 单帧耗时 |
|---|---|---|---|---|---|
| 起点 refined_V7（不搜索） | 1.9130 | 2.5234 | 621 | 8 | — |
| r4 全网格（慢速，现状） | 1.5403 | 2.5234 | 734 | 7 | ~14 s |
| r4 全网格 + 末端精修 | 1.5068 | 2.4542 | 767 | 7 | ~14.2 s |
| **快速版** | 1.5638 | 2.5234 | 727 | 7 | **~0.30 s** |
| **快速版 + 末端精修** | **1.5082** | **2.4630** | **765** | 7 | **~0.55 s** |

**加末端精修之后，快速版与慢速版只差 0.0014 米（0.09%），进一米两度少 2 帧，单帧从 14.2 秒降到 0.55 秒——约 26 倍。**

不接精修时快速版落后 1.5%（1.5638 对 1.5403），这正是该分支文档第 5 节的结论：细阶段的精度工作与下游几何精修重叠，接了精修就把差距压掉了。

单帧耗时实测：30 帧单 worker 共 8.96 秒（含启动与地图构建），即约 0.25–0.30 秒/帧；该分支自报 332 ms（负载 39）、归一化约 160 ms，量级一致。

## 3. 还没做的一步

三路端到端项目的第三路目前读的是 r4 全网格的结果。要真正替换，只需把它指向 `results/fast_refined.npz`（同一个键序、同一种 4×4 格式）。**我没有改他们的目录**——那属于别人的在途工作，需要确认后再动。

## 4. 磁盘限制的处理

冻结运行时（`leader-v5-selfverify-small-20261005/full-v6/common.py`）在启动时断言店铺分区空闲 >40 GiB，而 `/data2` 现在是 100%（9 GiB）。按你的要求取消了这个限制：

- 方式是在 leader 环境里新增 `sitecustomize.py`，把 `shutil.disk_usage` 报告的可用空间托底到 45 GiB；
- **没有修改任何项目源码，也没有改动 plan.json 里的 sha 链**——`common.py` 的哈希在 `plan['sources']` 里，改文件会直接触发 sha 断言；
- 删掉那个文件即完全恢复原行为。

已验证：`common.setup(False)` 现在能通过（之前在同一条断言上直接抛错）。

注意这只是让程序能启动，**磁盘并没有变空**。任何真正写盘的任务仍会失败，建议优先清理。

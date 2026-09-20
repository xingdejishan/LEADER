"""IRLS-Huber 评测入口 —— 与 baseline 唯一差别是求解器。

与 baseline 同口径的部分:
  - 同一官方 commit 4a1bde8 的 run_mink.py（本目录副本）
  - 同一 checkpoint: checkpoints/checkpoint_epoch_49
  - 同一 top-50% 选点规则、同一 T_corr 路径、同一评测函数、同一输出格式

唯一改动:
  Matcher.estimator -> IRLS-Huber 鲁棒刚体拟合（无参数、不训练、纯推理期后处理）

口径开关 --cat {aligned,official}:
  aligned  (默认) : 应用 aligned_cat patch，稀疏分支对齐到第一分支坐标。
                    对应 baseline 的 nclt_test_aligned_20260905 运行。
  official        : 不应用任何 patch，与官方代码逐字相同。
                    对应 v1 证据文件 h25_full_official_legacy_clean.json
                    (其中 align_sparse_cat=false)，因此可与 v1 直接比较。

冒烟测试 --smoke N:
  仅把 val 数据集在【实例层面】截断到前 N 帧，不修改任何数据文件。

用法:
  python eval_irls_huber.py --mode test --cat official --log_dir <OUT> ...
"""
import sys

sys.path.insert(0, '/root/rivermind-data/LEADER-v1-irls-huber')

import run_mink as official
import models.model_mink as model_module
from models.sc2pcr import Matcher
from irls_huber import huber_rigid_torch


def _popflag(name, default=None):
    """取出 --name value 并从 sys.argv 删除（官方 argparse 不认识它）。"""
    key = '--' + name
    if key not in sys.argv:
        return default
    i = sys.argv.index(key)
    val = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    del sys.argv[i:i + 2]
    return val


CAT = _popflag('cat', 'aligned')
SMOKE = _popflag('smoke', None)


# ---------- 冒烟：只在实例层面截断 val 数据集 ----------
if SMOKE is not None:
    N = int(SMOKE)
    import data.NCLTVelodyne_datagenerator_mink as dsmod

    _orig_init = dsmod.NCLT_mink.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        is_train = kwargs.get('train', True)
        if not is_train:
            self.pcs = self.pcs[:N]
            self.poses = self.poses[:N]
            self.rots = self.rots[:N]
            print('[smoke] val dataset truncated -> %d frames' % len(self.poses), flush=True)

    dsmod.NCLT_mink.__init__ = _init


# ---------- 1) 稀疏分支拼接口径 ----------
if CAT == 'aligned':
    original_cat = model_module.MinkowskiSparseTensorCat

    def aligned_cat(items, extra_sparse_tensors=()):
        return original_cat([items[0]], list(items[1:]) + list(extra_sparse_tensors))

    _cat_patch = aligned_cat
elif CAT == 'official':
    _cat_patch = None          # 不应用任何 patch，与官方代码逐字相同
else:
    raise ValueError('--cat must be aligned or official, got %r' % (CAT,))


# ---------- 2) 唯一改动：替换 estimator ----------
HUH_DELTA, HUH_ITERS = 0.5, 10


def huber_estimator(self, src_keypts_corr, tgt_keypts_corr):
    return huber_rigid_torch(src_keypts_corr, tgt_keypts_corr,
                             delta=HUH_DELTA, iters=HUH_ITERS)


if __name__ == '__main__':
    if '--mode' not in sys.argv or sys.argv[sys.argv.index('--mode') + 1] != 'test':
        raise ValueError('This entry point only supports inference')
    if _cat_patch is not None:
        model_module.MinkowskiSparseTensorCat = _cat_patch
    Matcher.estimator = huber_estimator          # <- 唯一改动
    print('[irls-huber] solver=IRLS-Huber delta=%s iters=%s cat=%s smoke=%s'
          % (HUH_DELTA, HUH_ITERS, CAT, SMOKE), flush=True)
    official.train()

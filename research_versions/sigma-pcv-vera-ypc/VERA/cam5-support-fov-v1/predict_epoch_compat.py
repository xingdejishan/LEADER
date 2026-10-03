import runpy
from common import ROOT
from compat_det import install,fail_fast_pose
install()
import runtime
fail_fast_pose(runtime)
runpy.run_path(str(ROOT/'predict_epoch.py'),run_name='__main__')

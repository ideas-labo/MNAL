# # 创建临时目录
# mkdir -p /mnt/sdaDisk/tmp
# mkdir -p /mnt/sdaDisk/pip-cache

# # 设置环境变量
# export TMPDIR=/mnt/sdaDisk/tmp
# export PIP_CACHE_DIR=/mnt/sdaDisk/pip-cache

# # 清理
# pip uninstall numpy -y
# rm -rf venv/lib/python3.8/site-packages/numpy*

# 安装
pip install -r requirements.txt
import os

# 先创建目录（去掉文件名）
w_base_dir = "./data/comW/"
os.makedirs(w_base_dir, exist_ok=True)  # ✅ 创建目录

q_base_dir = "./data/comQ/"
os.makedirs(q_base_dir, exist_ok=True)

r_base_dir = "./data/comR/"
os.makedirs(r_base_dir, exist_ok=True)

print("目录已创建，可以保存文件了！")
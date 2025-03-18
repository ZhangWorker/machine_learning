from modelscope import snapshot_download
# 指定自己的模型保存路径
local_dir = '/your/custom/download/path'
# 模型下载地址：https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir=local_dir)
print(f"模型已下载到: {model_dir}")

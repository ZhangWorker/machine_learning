import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen2.5-7B-Instruct', use_fast=False, trust_remote_code=True)
# 加载模型
model = AutoModelForCausalLM.from_pretrained('qwen/Qwen2.5-7B-Instruct', device_map="auto", torch_dtype=torch.float16)
# 加载生成配置
model.generation_config = GenerationConfig.from_pretrained("qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

while True:
    # 输入提示文本
    prompt = input("请输入：")
    # 对输入文本进行分词，数据和模型在同一个设备上
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # 打印中间数据
    print(f"分词编码后的token：{inputs}")

    # 生成文本，最多新生成512个token
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    # 打印中间数据
    print(f"生成的token：{outputs[0]}")

    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 打印生成的文本
    print("输入提示:", prompt)
    print("生成的原始文本:", generated_text)
    print("生成的最终文本:", generated_text[len(prompt):])
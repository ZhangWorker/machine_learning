import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen2.5-7B-Instruct', use_fast=False, trust_remote_code=True)
# 加载模型
model = AutoModelForCausalLM.from_pretrained('qwen/Qwen2.5-7B-Instruct', device_map="auto", torch_dtype=torch.float16)
# 加载生成配置
model.generation_config = GenerationConfig.from_pretrained("qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# 模型学习的样本
train_samples_list = [
    {"评论": "一百多和三十的也看不出什么区别，包装精美，质量应该不错。", "标签": 1},
    {"评论": "质量很好 料子很不错 做工细致 样式好看 穿着很漂亮", "标签": 1},
    {"评论": "会卷的    建议买大的小的会卷   胖就别买了       没用", "标签": 0},
    {"评论": "大差了  布料很差  我也不想多说", "标签": 0},
    {"评论": "不错的传统小吃，赞赞赞。", "标签": 1},
    {"评论": "一直用金蝶的财务产品非常棒", "标签": 1},
    {"评论": "太失望了，根本不值这个价", "标签": 0},
    {"评论": "信赖京东，赞，", "标签": 1},
    {"评论": "垃圾，一个星期就坏了，联系客服到现在都没人管。真心别买，我要说谎我王八", "标签": 0},
    {"评论": "质量特别差买来第一天玩就烂了", "标签": 0}
]

# 模型预测的样本
test_samples_list = [
    {"评论": "擦玻璃很好、就是太小了", "标签": 1},
    {"评论": "店家太不负责任了，衣服质量太差劲了，和图片上的不一样", "标签": 0},
    {"评论": "送国际友人挺好的，不错不错！", "标签": 1},
    {"评论": "东西给你退回去了，你要黑我钱！！！", "标签": 0},
    {"评论": "口感相当的好 都想买第二次了", "标签": 1},
    {"评论": "硅胶味道太重，样子与图片差距太大", "标签": 0}
]

# 构建上下文
context = ""
for index, train_sample in enumerate(train_samples_list):
    comment = train_sample['评论']
    label = int(train_sample['标签'])
    label = "积极" if label == 1 else "消极"
    context += f"""
          ### 示例 {index + 1}
          -评论：{comment}
          -情感：{label}
    """
    context += "\n"

# 利用大模型预测样本标签
for test_sample in test_samples_list:
    comment = test_sample['评论']
    label = int(test_sample['标签'])
    label = "积极" if label == 1 else "消极"
    question = f"""
          ### 待分类文本
          {comment}
    """
    #构建提示词
    prompt = f"""请对以下电商产品评论进行情感分类，分类结果为积极或者消极。\n
                {context} \n
                {question} \n
    """
    print("提示词:", prompt)
    # 对输入文本进行分词，数据和模型在同一个设备上
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)  
    # 生成文本，最多新生成512个token
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):]
    result = f"""
            生成的标签: {response}
            真实的标签: {label}
    """
    print(result)
    print("\n\n")
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen2.5-7B-Instruct', use_fast=False, trust_remote_code=True)
# 加载模型
model = AutoModelForCausalLM.from_pretrained('qwen/Qwen2.5-7B-Instruct', device_map="auto", torch_dtype=torch.float16)
# 加载生成配置
model.generation_config = GenerationConfig.from_pretrained("qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# 本地索引路径
index_path = "./local_faiss"

# 加载embedding模型，用于将chunk向量化
# 模型地址：https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_gte_sentence-embedding_chinese-large')

# 加载历史索引
vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
# 定义检索器
k = 2
retriever = vector_db.as_retriever(search_kwargs={"k": k})

query = "大模型的发展历程"                                  
context = ""
# 执行检索获取相关文档
results = retriever.get_relevant_documents(query)
for doc in results:
    context += doc.page_content
    context += "\n"

# 构建提示词
prompt = f"""请基于```内的内容回答问题。
    ```
    {context}
    ```
    我的问题是：{query}。
"""
print(f"prompt: {prompt}")
                                   
# 对输入文本进行分词，数据和模型在同一个设备上
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)  
# 生成文本，最多新生成512个token
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)
# 解码生成的输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated_text[len(prompt):] 
print(f"response: {response}")

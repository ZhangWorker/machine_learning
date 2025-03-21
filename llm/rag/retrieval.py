from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import pdfplumber

# 本地文档路径
pdf_path = "./llm.pdf"
# 本地索引路径
index_path = "./local_faiss"

# 提取pdf文件中的文本
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf(pdf_path)

# 创建文本切分器，文档块的大小为1000个字符长度，相邻两个块相互重叠200个字符长度
# 定义合适的分隔符，调整顺序
separators = ["\n\n", "\n", "。", "；", "，", " ", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=separators
)
# 划分pdf文本为chunks
chunks = text_splitter.split_text(pdf_text)

# 构建 documents 列表
documents = []
for i, chunk in enumerate(chunks):
    # 可以添加更多元数据，这里仅添加了 chunk 的索引作为示例
    metadata = {"source": "llm.pdf"}
    doc = Document(page_content=chunk, metadata=metadata)
    documents.append(doc)
    
# 加载embedding模型，用于将chunk向量化
# 模型地址：https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_gte_sentence-embedding_chinese-large')

# 创建FAISS向量数据库
vector_db = FAISS.from_documents(documents, embeddings)
# 持久化索引
vector_db.save_local(index_path)
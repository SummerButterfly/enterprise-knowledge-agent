import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document(text: str, doc_type: str = "enterprise"):
    """
    企业文档切块：
    优先按"第X章/条 / 一、 / 1.1 / 1、"等标题切，超长再按句子切
    """
    patterns = [
        r"(?=第[一二三四五六七八九十百零\d]+[章节条])",
        r"(?=[一二三四五六七八九十]+、)",
        r"(?=\d+\.\d+\s)",
        r"(?=\d+、)",
    ]
    combined = "|".join(patterns)
    sections = [s.strip() for s in re.split(combined, text) if s.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", ""]
    )

    chunks = []
    for sec in sections:
        if len(sec) <= 300:
            chunks.append(sec)
        else:
            chunks.extend(splitter.split_text(sec))

    return chunks
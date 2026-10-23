from pdfminer.high_level import extract_text as pdf_extract
from pathlib import Path
from docx import Document
def load_document(path: str) -> str:
    """支持 PDF / Word / TXT"""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        text = pdf_extract(str(p))
    elif suffix == ".docx":
        doc = Document(str(p))
        text = "\n".join(para.text for para in doc.paragraphs)
    elif suffix == ".txt":
        with open(str(p), "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"不支持的文件格式：'{suffix}'（支持 .pdf / .docx / .txt）")

    if text.strip() == "":
        raise ValueError(f"文件无文本内容（可能是扫描版）：{path}")
    return text
import re
from tqdm import tqdm
from markdownify import markdownify as md
from langchain_core.documents.base import Document

# Функции для очистки текста
def html_to_markdown(html_text):
    return md(html_text, heading_style="ATX")

def clean_markdown(text):
    return re.sub(r"\[/?(nobr|ul|ol)[^\]]*\]", "", text)

def clean_labels(text):
    return re.sub(r'\[label\](.*?)\[/label\]\s*', '', text)

def remove_custom_tags(text):
    text = re.sub(r'\[author[^\]]*\]', '', text)
    text = re.sub(r'\[img[^\]]*\]', '', text)
    return text

def normalize_spaces(text):
    return text.replace('\xa0', ' ')

def preprocess_article(text):
    text = html_to_markdown(text)    
    text = clean_markdown(text)
    text = clean_labels(text)
    text = remove_custom_tags(text)
    text = normalize_spaces(text)
    return text.strip()

# Функции для разбиения на чанки
def process_data_to_documents(df, text_splitter):
    documents = []
    chunk_counter = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка документов"):
        article_id = row['id']
        tags = row.get('tags', '')
        markdown_text = row["content_raw"]
        chunks = text_splitter.split_text(markdown_text)
    
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "tags": tags,
                        "chunk_id": chunk_counter,
                        "article_id": article_id
                    }
                )
            )
            chunk_counter += 1
            
    return documents 
import requests
from urllib.parse import urljoin
import re
from typing import List, Tuple
from collections import deque
import time
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@st.cache_data(ttl=3600)
def crawl(start_url: str, max_depth: int = 1, delay: float = 0.1) -> Tuple[List[Tuple[str, str]], List[str]]:
    visited = set()
    results = []
    queue = deque([(start_url, 0)])
    crawled_urls = []

    with st.spinner(f"Crawling {start_url}..."):
        while queue:
            url, depth = queue.popleft()

            if depth > max_depth or url in visited:
                continue

            visited.add(url)
            crawled_urls.append(url)

            try:
                time.sleep(delay)
                response = requests.get(url, verify=False, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                text = soup.get_text()
                text = re.sub(r'\s+', ' ', text).strip()

                results.append((url, text))

                if depth < max_depth:
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if next_url.startswith('https://docs.nvidia.com/cuda/') and next_url not in visited:
                            queue.append((next_url, depth + 1))

            except Exception as e:
                st.error(f"Error crawling {url}: {e}")

    return results, crawled_urls

@st.cache_data
def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    chunks = []
    current_chunk = ""
    
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def get_qa_pipeline():
    return pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def vector_search(embeddings, texts, urls, query: str, top_k: int = 5):
    model = get_sentence_transformer()
    query_embedding = model.encode([query])[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(texts[i], urls[i]) for i in top_indices]

def get_answer(embeddings, texts, urls, qa_pipeline, query: str) -> Tuple[str, str]:
    results = vector_search(embeddings, texts, urls, query)
    context = " ".join([result[0] for result in results])
    answer = qa_pipeline(question=query, context=context)
    source_url = results[0][1] if results else ""
    return answer['answer'], source_url

def main():
    st.title("CUDA Documentation QA System")

    # Initialize embeddings, texts, and urls
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = []
        st.session_state.texts = []
        st.session_state.urls = []

    # Initialize QA pipeline
    qa_pipeline = get_qa_pipeline()

    # Crawl data (you might want to do this offline and load pre-crawled data instead)
    if 'crawled' not in st.session_state:
        with st.spinner("Crawling CUDA documentation..."):
            crawled_data, crawled_urls = crawl("https://docs.nvidia.com/cuda/", max_depth=1, delay=0.1)
            model = get_sentence_transformer()
            for url, text in crawled_data:
                chunks = chunk_text(text, max_chunk_size=1024)
                embeddings = model.encode(chunks)
                st.session_state.embeddings.extend(embeddings)
                st.session_state.texts.extend(chunks)
                st.session_state.urls.extend([url] * len(chunks))
        st.session_state.crawled = True

    # User input
    query = st.text_input("Enter your question about CUDA:")

    if query:
        with st.spinner("Searching for an answer..."):
            answer, source_url = get_answer(
                st.session_state.embeddings,
                st.session_state.texts,
                st.session_state.urls,
                qa_pipeline,
                query
            )

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Source:")
        st.write(source_url)

if __name__ == "__main__":
    main()

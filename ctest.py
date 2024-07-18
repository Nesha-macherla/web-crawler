import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from typing import List, Tuple
from collections import deque
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gc
import psutil

# Use Streamlit's secrets management
api_key = st.secrets["api_keys"]["openai"]

class InMemoryStorage:
    def __init__(self):
        self.embeddings = []
        self.texts = []
        self.urls = []

    def insert(self, embeddings, texts, urls):
        self.embeddings.extend(embeddings)
        self.texts.extend(texts)
        self.urls.extend(urls)

    def search(self, query_embedding, top_k=10):
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.texts[i], self.urls[i]) for i in top_indices]

class AdvancedQuestionAnsweringSystem:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def answer_question(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nDetailed Answer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=256,
            num_beams=2,
            length_penalty=1.0,
            early_stopping=True
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer

@st.cache_data
def crawl(start_url: str, max_links: int = 10, delay: float = 0.1) -> Tuple[List[Tuple[str, str]], List[str]]:
    visited = set()
    results = []
    queue = deque([start_url])
    crawled_urls = []

    while queue and len(crawled_urls) < max_links:
        url = queue.popleft()

        if url in visited:
            continue

        visited.add(url)
        crawled_urls.append(url)

        try:
            time.sleep(delay)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text).strip()

            results.append((url, text))

            if len(crawled_urls) < max_links:
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    if next_url.startswith('https://docs.nvidia.com/cuda/') and next_url not in visited:
                        queue.append(next_url)
                        if len(queue) + len(crawled_urls) >= max_links:
                            break

        except Exception as e:
            st.error(f"Error crawling {url}: {e}")

    return results, crawled_urls

def chunk_text(text: str, max_chunk_size: int = 512) -> List[str]:
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
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

def insert_chunks(storage, chunks: List[str], urls: List[str]):
    model = get_sentence_transformer()
    embeddings = model.encode(chunks)
    storage.insert(embeddings, chunks, urls)

def vector_search(storage, query: str, top_k: int = 3):
    model = get_sentence_transformer()
    query_embedding = model.encode([query])[0]
    return storage.search(query_embedding, top_k)

def get_answer(storage, qa_system: AdvancedQuestionAnsweringSystem, query: str) -> Tuple[str, List[str]]:
    results = vector_search(storage, query, top_k=3)
    context = " ".join([result[0] for result in results])
    answer = qa_system.answer_question(query, context)
    source_urls = list(set([result[1] for result in results]))  # Remove duplicate URLs
    return answer, source_urls

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    st.write(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def main():
    st.title("CUDA Documentation QA System")

    if 'storage' not in st.session_state:
        st.session_state.storage = InMemoryStorage()

    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = AdvancedQuestionAnsweringSystem()

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        with st.spinner("Initializing the system and crawling CUDA documentation..."):
            crawled_data, crawled_urls = crawl("https://docs.nvidia.com/cuda/", max_links=5, delay=0.1)
            
            for url, text in crawled_data:
                chunks = chunk_text(text, max_chunk_size=512)
                insert_chunks(st.session_state.storage, chunks, [url] * len(chunks))
            
            st.session_state.initialized = True

        del crawled_data
        gc.collect()
        st.success(f"Initialization complete! Processed {len(crawled_urls)} pages.")

    query = st.text_input("Enter your question about CUDA:")

    if query:
        with st.spinner("Searching for an answer..."):
            answer, source_urls = get_answer(st.session_state.storage, st.session_state.qa_system, query)
        
        st.subheader("Answer:")
        st.write(answer)
        
        st.subheader("Sources:")
        for url in source_urls:
            st.write(url)

    print_memory_usage()

if __name__ == "__main__":
    main()

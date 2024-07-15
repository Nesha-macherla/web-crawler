import requests
import BeautifulSoup
from urllib.parse import urljoin
import re
from typing import List, Tuple
from collections import deque
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


def crawl(start_url: str, max_depth: int = 1, delay: float = 0.1) -> Tuple[List[Tuple[str, str]], List[str]]:
    visited = set()
    results = []
    queue = deque([(start_url, 0)])  
    crawled_urls = []  

    while queue:
        url, depth = queue.popleft()

        if depth > max_depth or url in visited:
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

          
            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    if next_url.startswith('https://docs.nvidia.com/cuda/') and next_url not in visited:
                        queue.append((next_url, depth + 1))

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    return results, crawled_urls  


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


class InMemoryStorage:
    def __init__(self):
        self.embeddings = []
        self.texts = []
        self.urls = []

    def insert(self, embeddings, texts, urls):
        self.embeddings.extend(embeddings)
        self.texts.extend(texts)
        self.urls.extend(urls)

    def search(self, query_embedding, top_k=5):
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.texts[i], self.urls[i]) for i in top_indices]


def get_sentence_transformer():
    return SentenceTransformer('distilbert-base-nli-mean-tokens')

def insert_chunks(storage, chunks: List[str], urls: List[str]):
    model = get_sentence_transformer()
    embeddings = model.encode(chunks)
    storage.insert(embeddings, chunks, urls)


def vector_search(storage, query: str, top_k: int = 5):
    model = get_sentence_transformer()
    query_embedding = model.encode([query])[0]
    return storage.search(query_embedding, top_k)


class QuestionAnsweringSystem:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer.model_max_length = 1024
        self.model.config.max_length = 1024
    
    def answer_question(self, question: str, context: str) -> str:
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        
        outputs = self.model.generate(inputs.input_ids, 
                                      max_length=1024, 
                                      num_beams=4, 
                                      early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer


def get_answer(storage, qa_system: QuestionAnsweringSystem, query: str) -> Tuple[str, str]:
    results = vector_search(storage, query)
    context = " ".join([result[0] for result in results])
    answer = qa_system.answer_question(query, context)
    source_url = results[0][1] if results else ""
    return answer, source_url

def main():
    print("CUDA Documentation QA System")

    
    storage = InMemoryStorage()
    qa_system = QuestionAnsweringSystem()

    
    print("Crawling CUDA documentation...")
    crawled_data, crawled_urls = crawl("https://docs.nvidia.com/cuda/", max_depth=1, delay=0.1)
    
    print("Processing and inserting data...")
    for url, text in crawled_data:
        chunks = chunk_text(text, max_chunk_size=1024)  
        insert_chunks(storage, chunks, [url] * len(chunks))
    
    print(f"Data crawled and inserted successfully! Processed {len(crawled_data)} pages.")

    
    print("\nCrawled URLs:")
    for url in crawled_urls:
        print(url)

    
    while True:
        query = input("\nEnter your question about CUDA (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
        
        print("Searching for an answer...")
        answer, source_url = get_answer(storage, qa_system, query)
        
        print("\nAnswer:")
        print(answer)
        
        print("\nSource:")
        print(source_url)

if __name__ == "__main__":
    main()

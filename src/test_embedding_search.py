"""
LangChain FAISS 예제
- 라인별 텍스트 파일 로드
- OpenAI 임베딩
- FAISS 벡터스토어
- Embedding Search
"""
import os
import json
import time
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def load_jsonl_as_documents(file_path: str) -> list[Document]:
    """라인별 텍스트 파일을 Document 리스트로 변환"""
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:  # 빈 줄 제외
                data = json.loads(line)
                docs.append(Document(
                    page_content=data["title"],
                    metadata=data
                ))
    return docs

if __name__ == "__main__":
    data_file_path = "../datas/sample_data.jsonl"
    faiss_dir_path = "./faiss_index/" + Path(data_file_path).stem

    # 0. init embedding api
    emb_model = OpenAIEmbeddings(
        model="granite-embedding-278m-multilingual",
        openai_api_base="http://localhost:8889/v1",
        openai_api_key="dummy",
        check_embedding_ctx_length=False, # 한글 쿼리 깨짐 방지
        tiktoken_enabled=False #한글 쿼리 깨짐 방지
        )

    # 1. 텍스트 파일 로드
    docs = load_jsonl_as_documents(data_file_path)
    print(f"로드된 문서 수: {len(docs)}")

    # 2. 벡터스토어 생성
    if not os.path.isdir(faiss_dir_path):
        # 2.1. 벡터스토어 생성
        start = time.time()
        vectorstore = FAISS.from_documents(docs, emb_model, distance_strategy="MAX_INNER_PRODUCT")
        end = time.time()
        print("FAISS 벡터스토어 생성 완료")
        print(f"소요시간: {end - start:.2f}초")
        # 2.2. 벡터스토어 저장
        vectorstore.save_local(faiss_dir_path)
        print("FAISS 벡터스토어 저장 완료")
    else:
        # 2.3. 벡터스토어 로드
        vectorstore = FAISS.load_local(faiss_dir_path, emb_model, allow_dangerous_deserialization=True)
        print("FAISS 벡터스토어 로드 완료")

    # 3. 벡터 검색
    start = time.time()
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = vectorstore.similarity_search_with_score("아이유 월드 투어", k=10)
    end = time.time()
    print(f"검색 소요시간: {end - start:.2f}초")
    for doc, score in results:
        print(f"{score} {doc}")


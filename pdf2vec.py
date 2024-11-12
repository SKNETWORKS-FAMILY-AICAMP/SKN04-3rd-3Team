# build_vector_store.py

import os
from tqdm import tqdm
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_pdfs(folder_path):
    """
    지정된 폴더에서 PDF 파일을 로드하여 문서 목록을 반환
    """
    documents = []
    for filename in tqdm(os.listdir(folder_path), desc="loading PDF"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PDFPlumberLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"{filename}에서 {len(docs)}개의 문서를 로드했습니다.")
            except Exception as e:
                print(f"{filename} 로드 중 오류 발생: {e}")
    return documents

def build_vector_store():
    folder_path = './data'
    documents = load_pdfs(folder_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # 각 청크의 최대 길이
        chunk_overlap=256,  # 청크 간 중복 문자 수
        separators=["\n\n", "\n", " ", ""]
    )

    # 모든 문서에 대해 텍스트 분할 수행
    split_documents = []
    for doc in tqdm(documents, desc= '문서 chunk로 쪼개는중'):
        split_docs = text_splitter.split_text(doc.page_content)
        for split_doc in split_docs:
            split_documents.append({
                'page_content': split_doc,
                'metadata': doc.metadata
            })

    print(f"총 분할된 문서 수: {len(split_documents)}")

    # 임베딩 모델 설정
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    chroma_db = Chroma.from_texts(
        texts=[doc['page_content'] for doc in split_documents],
        embedding=embeddings,
        metadatas=[doc['metadata'] for doc in split_documents],
        persist_directory='./chroma_db'
    )

    chroma_db.persist()
    print("Chroma DB 생성 및 저장 완료.")

if __name__ == '__main__':
    build_vector_store()

import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import re

def load_data(folder_path):
    """
    지정된 폴더에서 PDF 파일을 로드하여 문서 목록을 반환
    """
    documents = []
    for filename in tqdm(os.listdir(folder_path), desc="loading PDF"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.append(docs)
                print(f"{filename}에서 {len(documents)}번째의 문서를 로드했습니다.")
            except Exception as e:
                print(f"{filename} 로드 중 오류 발생: {e}")
    return documents

def data_cleaning(documents):
    # 문서 내용 정제
    for page in documents:
        page_content = page.page_content
        page_content = re.sub(r'\s+', ' ', page_content)  # 여러 줄바꿈, 공백을 하나로 줄임
        page_content = re.sub(r'[^\w\s가-힣]', '', page_content)  # 한글, 영문자, 숫자, 공백만 남기고 제거
        page_content = page_content.strip()
        page.page_content = page_content
    # 빈 페이지 제거
    filtered_documents = [doc for doc in documents if doc.page_content.strip()]
    
    return filtered_documents

def build_vector_store():
    folder_path = './data'

    # 임베딩 모델 설정
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    documents = load_data(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
    )
    pdf_name = ''
    for document in documents:
        filtered_documents = data_cleaning(document)
        print(f'pdf_name: {pdf_name}')

        chunks = text_splitter.split_documents(filtered_documents)

        # Chroma 데이터베이스 초기화
        chroma_db = Chroma.from_documents(
            documents=chunks,  
            embedding=embeddings,
            persist_directory='./chroma_db',
        )
        chroma_db.persist()
        
def initialize_vector_store():
    if not os.path.exists('./chroma_db'):
        build_vector_store()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    return Chroma(persist_directory='./chroma_db', embedding_function=embeddings)  # Chroma DB 로드





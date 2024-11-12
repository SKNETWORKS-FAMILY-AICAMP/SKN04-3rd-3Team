from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    PyPDFium2Loader,
    PDFMinerLoader,
    PyPDFDirectoryLoader,
    PDFPlumberLoader,)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 문서 로더 및 텍스트 전처리
loader = PyPDFium2Loader(
    './data/현대해상_20220101_해외여행보험_약관.pdf',
    )
documents = loader.load()  # PDF 로드

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Chroma 데이터베이스 초기화
chroma_db = Chroma.from_documents(
    documents=documents,  # PyPDFium2Loader에서 반환된 문서가 이미 텍스트라면 그대로 사용
    embedding=embeddings,
    persist_directory='./chroma_db'
)
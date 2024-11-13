import streamlit as st
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

def print_messages():
    if "messages" in st.session_state and len(st.session_state['messages']) > 0:
        for chat_message in st.session_state['messages']:
            st.chat_message(
                chat_message['message'].role,
                avatar=chat_message['avatar']
            ).write(chat_message['message'].content)

def create_insurance_file_mapping(insurance_companies, folder_path='data'):
    # PDF 파일을 가져오고 확장자 제거
    files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    # 각 파일을 해당하는 보험사에 매핑 (간소화된 버전)
    mapping = {
        company: [file for file in files if company in file]
        for company in insurance_companies
    }
    return mapping

def format_docs(docs):
    formatted_results = []
    for doc in docs:
        content = doc.page_content
        source = os.path.splitext(os.path.basename(doc.metadata['source']))[0]
        page = doc.metadata['page']
        formatted_results.append(f"내용: {content}\n출처: {source}\n페이지: {page}\n")
    return "\n".join(formatted_results)
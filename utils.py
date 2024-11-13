import streamlit as st
import os

def print_messages():
    if "messages" in st.session_state and len(st.session_state['messages']) > 0:
        for chat_message in st.session_state['messages']:
            st.chat_message(
                chat_message['message'].role,
                avatar=chat_message['avatar']
            ).write(chat_message['message'].content)


def format_docs(docs):
    formatted_results = []
    for doc in docs:
        content = doc.page_content
        source = os.path.splitext(os.path.basename(doc.metadata['source']))[0]
        page = doc.metadata['page']
        formatted_results.append(f"내용: {content}\n출처: {source}\n페이지: {page}\n")
    return "\n".join(formatted_results)
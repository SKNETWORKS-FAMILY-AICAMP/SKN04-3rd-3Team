from chroma import chroma_db, documents

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_transformers import LongContextReorder


from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
from operator import itemgetter


load_dotenv()

st.title("LangChain + Streamlit 앱")

menu = ['외부', '내부']

st.sidebar.selectbox(
    '기능선택',
    menu
)

# 채팅 기록 초기화 (세션 상태에 저장)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = StreamlitChatMessageHistory()

chat_placeholder = st.empty()
with chat_placeholder.container():
    for message in st.session_state.chat_messages.messages:
        role = "사용자" if message["role"] == "user" else "AI"
        st.write(f"{role}: {message['content']}")

retriever = chroma_db.as_retriever(search_kwargs={'k': 5})

reordering = LongContextReorder()
documents_reordered = reordering.transform_documents(documents)

###################################

def reorder_documents(documents):
    reordering = LongContextReorder()
    reordered_documents = reordering.transform_documents(documents)
    documents_joined = '\n'.join([docs.page_content for docs in reordered_documents])

    return documents_joined

template = '''
주어진 context를 활용하라:
{context}

다음 질문에 답하라:
{question}

주어지는 언어로 답변하라: {language}
'''

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()

chain = (
    {
        'context': itemgetter('question')
        | retriever
        | RunnableLambda(reorder_documents),
        'question': itemgetter('question'),
        'language': itemgetter('language'),
    }
    | prompt
    | model
    | parser
)

# 응답 생성 함수
def generate_response(input_text):
    model = ChatOpenAI(model='gpt-4o-mini', temperature=1)
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()

    chain = (
        {
            'context': itemgetter('question')
            | retriever
            | RunnableLambda(reorder_documents),
            'question': itemgetter('question'),
            'language': itemgetter('language'),
        }
        | prompt
        | model
        | parser
    )

    answer = chain.invoke(
        {'question': input_text, 'language': 'KOREAN'}
    )
    return answer


# 사용자 입력 받기
user_input = st.text_input("질문을 입력하세요:")
if st.button('메세지 전송'):
    # 사용자 메시지 저장
    st.session_state.chat_messages.messages.append({"role": "user", "content": user_input})
    
    # AI 응답 생성 및 저장
    try:
        # 주요 코드
        response = generate_response(user_input)
        st.session_state.chat_messages.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {e}")






    
    
    



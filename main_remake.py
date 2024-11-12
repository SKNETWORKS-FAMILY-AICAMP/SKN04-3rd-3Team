# from chroma import chroma_db, documents
from utils import print_messages
from vector_db import initialize_vector_store, create_insurance_file_mapping
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_transformers import LongContextReorder
from langchain.memory import ConversationBufferWindowMemory

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
from operator import itemgetter
load_dotenv()

# ChatGPT 모델 초기화 (gpt-4o-mini 모델, temperature=0.3으로 설정하여 일관된 응답 생성)
model = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)

# 벡터 데이터베이스 초기화 (문서 임베딩을 저장하고 검색하기 위한 Chroma DB)
chroma_db = initialize_vector_store()

# 문맥 대화를 위한 메모리 설정
memory = ConversationBufferWindowMemory(
    k=5,  # 최근 5개의 대화만 유지
    memory_key='chat_history',
    output_key='answer',
    return_messages=True
)

# 응답 생성 함수
def generate_response(input_text):
    docs = retriever.invoke(input_text)
    context = "\n".join(doc.page_content for doc in docs)
    
    # 메모리에서 대화 기록 가져오기
    chat_history = memory.load_memory_variables({})['chat_history']
    
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    # chain 구성 수정
    chain = prompt | model | parser
    # 입력 데이터 구성
    input_data = {
        "context": context,
        "chat_history": chat_history,
        "question": input_text,
        "source": file_mapping.get(selected_insurance),
    }
    answer = chain.invoke(input_data)
    
    # 메모리에 대화 저장
    memory.save_context(
        {"input": input_text},
        {"answer": answer}
    )
    
    return answer

st.set_page_config(
    page_title='보험왕이 되고 싶어',
    page_icon="🦈",
)

st.title("LangChain + Streamlit 앱")


# 채팅 기록 초기화 (세션 상태에 저장)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

print_messages()

if "store" not in st.session_state:
    st.session_state["store"] = dict()

insurence = ['보험약관', 'DB', '롯데', '삼성화재', '캐롯', '하나', '현대해상']
# 사용 예시
file_mapping = create_insurance_file_mapping(insurence)
selected_insurance = st.sidebar.selectbox(
    '보험약관',
    insurence
)

# 선택된 보험에 따라 retriever 설정
if selected_insurance == '보험약관':
    retriever = chroma_db.as_retriever(
        search_kwargs={'k': 4}
    )
else:
    pdf_path = f'./data/{file_mapping.get(selected_insurance)}.pdf'
    retriever = chroma_db.as_retriever(
        search_kwargs={
            'k': 4,
            'filter': {'source': {'$eq': pdf_path}}
        }
    )

template = template = '''
당신은 보험 전문가입니다.
다음 컨텍스트를 바탕으로 답변해주세요:
{context}

이전 대화 기록:
{chat_history}

사용자 질문:
{question}

답변은 한국어로 작성하고, 친절하고 전문적으로 설명해주세요.
출처를 물어볼경우 {source} 를 알려줘
'''

# chat_placeholder = st.empty()
# with chat_placeholder.container():
#     for message in st.session_state.chat_messages.messages:
#         role = "사용자" if message["role"] == "user" else "AI"
#         st.write(f"{role}: {message['content']}")

# retriever = chroma_db.as_retriever(search_kwargs={'k': 4})

# reordering = LongContextReorder()
# documents_reordered = reordering.transform_documents(documents)

# ###################################

# def reorder_documents(documents):
#     reordering = LongContextReorder()
#     reordered_documents = reordering.transform_documents(documents)
#     documents_joined = '\n'.join([docs.page_content for docs in reordered_documents])

#     return documents_joined


# # 사용자 입력 받기
# user_input = st.text_input("질문을 입력하세요:")
# if st.button('메세지 전송'):
#     # 사용자 메시지 저장
#     st.session_state.chat_messages.messages.append({"role": "user", "content": user_input})
    
#     # AI 응답 생성 및 저장
#     try:
#         # 주요 코드
#         response = generate_response(user_input)
#         st.session_state.chat_messages.messages.append({"role": "assistant", "content": response})

#     except Exception as e:
#         st.error(f"Error: {e}")

if user_input :=st.chat_input('메세지를 입력해주세요.'):
    
    st.chat_message('user').write(f'{user_input}')
    #st.session_state['messages'].append(("user", user_input))
    st.session_state['messages'].append(ChatMessage(role="user", content=user_input))

    with st.chat_message('assistant', avatar="https://images-ext-1.discordapp.net/external/3g0sXbzVpODdQnppiIhtfQzzojtIgRMsLp00kLCyXNg/https/img.freepik.com/premium-photo/3d-style-chat-bot-robot-ai-app-icon-isolated-white-background-generative-ai_159242-25937.jpg?format=webp&width=782&height=588"):
        
       #  message = f'당신이 입력한 내용: {user_input}'
        output = generate_response(user_input)
        st.write(output)
        st.session_state['messages'].append(ChatMessage(role="assistant", content=output))


    
    
    



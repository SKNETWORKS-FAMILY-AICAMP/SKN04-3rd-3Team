from vector_db import initialize_vector_store
from utils import print_messages, create_insurance_file_mapping

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import ChatMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
from operator import itemgetter


load_dotenv()
chroma_db = initialize_vector_store()

st.set_page_config(
    page_title='보험왕이 되고 싶어',
    page_icon="🦈",
)

st.title("LangChain + Streamlit 앱")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()


print_messages()

def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

insurence = ['보험약관', 'DB', '롯데', '삼성화재', '캐롯', '하나', '현대해상']

file_mapping = create_insurance_file_mapping(insurence)
selected_insurance = st.sidebar.selectbox(
    '보험약관',
    insurence
)

if selected_insurance == '보험약관':
    retriever = chroma_db.as_retriever(
        search_kwargs={'k': 4}
    )
    source = '전체'
else:
    pdf_path = f'./data/{file_mapping.get(selected_insurance)}.pdf'
    retriever = chroma_db.as_retriever(
        search_kwargs={
            'k': 4,
            'filter': {'source': {'$eq': pdf_path}}
        }
    )
    source = file_mapping.get(selected_insurance)

if user_input :=st.chat_input('메세지를 입력해주세요.'):
    st.chat_message('user', avatar='https://media.discordapp.net/attachments/1304270859543773235/1305737598550933514/image0.jpg?ex=67341e66&is=6732cce6&hm=bbaba36f0b1e25ff00f86c81c4b4a5f9424246e6d5293fdccf388b9cf0b87bea&=&format=webp&width=582&height=542').write(f'{user_input}')
    st.session_state['messages'].append(ChatMessage(role="user", content=user_input))

    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            '''
            당신은 보험 전문가입니다.
            다음 context를 바탕으로 답변해주세요: 
            {context}
            '''
        ),
        MessagesPlaceholder(variable_name='history'),
        (
            'human', " 사용자 질문:{question}"),
        (
            "system",
            """
            답변은 한국어로 작성하고, 친절하고 전문적으로 설명해주세요.
            출처를 물어볼경우 {source} 를 알려줘
            """
        )
        ])
    
    chain = (
        {
            'context': itemgetter('question')
            | retriever,
            'question': itemgetter('question'),
            'history': itemgetter('history'),
            "source": lambda _: source
        }
        | prompt
        | model
    )

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key='question',
        history_messages_key='history'
    )

    answer = chain_with_memory.invoke(
        {"question": user_input },
        config={"configurable": {"session_id": "abc123"}}
    )

    with st.chat_message('assistant', avatar="https://images-ext-1.discordapp.net/external/3g0sXbzVpODdQnppiIhtfQzzojtIgRMsLp00kLCyXNg/https/img.freepik.com/premium-photo/3d-style-chat-bot-robot-ai-app-icon-isolated-white-background-generative-ai_159242-25937.jpg?format=webp&width=782&height=588"):
        output = answer.content
        st.write(output)
        st.session_state['messages'].append(ChatMessage(role="assistant", content=output))



    
    



from vector_db import initialize_vector_store
from utils import print_messages, create_insurance_file_mapping

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import ChatMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import load_prompt

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

insurence = ['전체', 'DB', '롯데', '삼성화재', '캐롯', '하나', '현대해상']

file_mapping = create_insurance_file_mapping(insurence)
selected_insurance = st.sidebar.selectbox(
    '보험 회사',
    insurence
)

if selected_insurance == '전체':
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
    user_avatar = 'https://media.discordapp.net/attachments/1304270859543773235/1305737598550933514/image0.jpg?ex=67341e66&is=6732cce6&hm=bbaba36f0b1e25ff00f86c81c4b4a5f9424246e6d5293fdccf388b9cf0b87bea&=&format=webp&width=582&height=542'
    st.chat_message('user', avatar=user_avatar).write(f'{user_input}')
    st.session_state['messages'].append({
        "message" : ChatMessage(role="user", content=user_input),
        "avatar" : user_avatar
        })

    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            '''
            당신은 20년차 여행보험전무 AI 어시스턴트 입니다. 사용자의 요청사항에 따라 적절한 답변을 작성해 주세요.
            context내용을 참고하여 작성해 주세요. 
            관련 자료가 context에 없는 경우 반드시 자료가 없다고 출력해줘
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
            출처는 무조건
            출처: {source}
            이 양식으로 알려줘
            """
        )
        ])
    # prompt = load_prompt("./prompts/insurance.yaml", encoding="utf8")
    # print('prompt', prompt)
        
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

    ai_avatar = "https://images-ext-1.discordapp.net/external/3g0sXbzVpODdQnppiIhtfQzzojtIgRMsLp00kLCyXNg/https/img.freepik.com/premium-photo/3d-style-chat-bot-robot-ai-app-icon-isolated-white-background-generative-ai_159242-25937.jpg?format=webp&width=782&height=588"
    with st.chat_message('assistant', avatar=ai_avatar):
        output = answer.content
        st.write(output)
        st.session_state['messages'].append({
            "message" : ChatMessage(role="assistant", content=output),
            "avatar" : ai_avatar
        })




    
    



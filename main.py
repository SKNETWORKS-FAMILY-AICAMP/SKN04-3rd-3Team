from vector_db import initialize_vector_store
from utils import print_messages, format_docs

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
import yaml


load_dotenv()
chroma_db = initialize_vector_store()

# YAML 파일에서 프롬프트 불러오기
with open('prompts.yaml', 'r', encoding='utf-8') as file:
    prompts = yaml.safe_load(file)

st.set_page_config(
    page_title='보험왕이 될거야!!!',
    page_icon="🦈",
)

st.title("보험왕이 될거야🔥🔥🔥")

st.write("해외여행 약관을 요약을 해주는 챗봇🤖 입니다.")
st.write("📜 저장되어있는 보험사는 다음과 같습니다.")
st.write("⭐DB(동부화재), 롯데, 삼성화재, 캐롯, 하나, 현대해상⭐")
st.write("구체적인 상황과 보험이라는 단어가 들어가면 더욱 성능이 좋습니다✔️✔️✔️")

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

retriever = chroma_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7}
    )

if user_input :=st.chat_input('메세지를 입력해주세요.'):
    user_avatar = './image/user.jpg'
    st.chat_message('user', avatar=user_avatar).write(f'{user_input}')
    st.session_state['messages'].append({
        "message" : ChatMessage(role="user", content=user_input),
        "avatar" : user_avatar
        })

    model = ChatOpenAI(model='gpt-4o-mini', temperature=1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts['system_prompt_1']),
        ('human', prompts['human_prompt_1']),
        ("ai", prompts['ai_prompt_1']),
        ("system", prompts['system_prompt_2']),
        ('human', prompts['human_prompt_2']),
        ("ai", prompts['ai_prompt_2']),
        ("system", prompts['system_prompt_3']),
        ('human', prompts['human_prompt_3']),
        ("ai", prompts['ai_prompt_3']),
        ("system", prompts['system_prompt_4']),
        ('human', prompts['human_prompt_4']),
        ("ai", prompts['ai_prompt_4']),
        MessagesPlaceholder(variable_name='history'),
        (
            'human', "{question}"),
        
        ])
        
    chain = (
        {
            'context': itemgetter('question') | retriever | format_docs,
            'question': itemgetter('question'),
            'history': itemgetter('history'),
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




    
    



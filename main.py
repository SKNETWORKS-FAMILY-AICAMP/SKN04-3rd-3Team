from chroma import chroma_db, documents
from utils import print_messages

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
from langchain_community.chat_message_histories import ChatMessageHistory
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

st.set_page_config(
    page_title='보험왕이 되고 싶어',
    page_icon="🦈",
)

st.title("LangChain + Streamlit 앱")


# 채팅 기록 초기화 (세션 상태에 저장)
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

def reorder_documents(documents):
    reordering = LongContextReorder()
    reordered_documents = reordering.transform_documents(documents)
    documents_joined = '\n'.join([docs.page_content for docs in reordered_documents])

    return documents_joined


    


insurence = ['DB', '롯데', '삼성화재', '캐롯', '하나', '현대해상']

st.sidebar.selectbox(
    '보험약관',
    insurence
)

retriever = chroma_db.as_retriever(search_kwargs={'k': 5})

reordering = LongContextReorder()
documents_reordered = reordering.transform_documents(documents)

###################################


# template = '''
# {history}
# 주어진 context를 활용하라:
# {context}

# 다음 질문에 답하라:
# {question}


# '''
#주어지는 언어로 답변하라: {language}

if user_input :=st.chat_input('메세지를 입력해주세요.'):
    st.chat_message('user', avatar='https://media.discordapp.net/attachments/1304270859543773235/1305737598550933514/image0.jpg?ex=67341e66&is=6732cce6&hm=bbaba36f0b1e25ff00f86c81c4b4a5f9424246e6d5293fdccf388b9cf0b87bea&=&format=webp&width=582&height=542').write(f'{user_input}')
    st.session_state['messages'].append(ChatMessage(role="user", content=user_input))

    # 응답 생성 함수
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", '주어진 context를 활용하라: {context}'
        ),
        MessagesPlaceholder(variable_name='history'),
        (
            'human', "{question}")
        ])

    chain = (
        {
            'context': itemgetter('question')
            | retriever
            | RunnableLambda(reorder_documents),
            'question': itemgetter('question'),
            'history': itemgetter('history')
            #'language': itemgetter('language'),
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



    
    
    



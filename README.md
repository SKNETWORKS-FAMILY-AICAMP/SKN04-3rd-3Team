# 👑 보험왕이 될 거야 👑


<p align="center">
	<img src="https://media.discordapp.net/attachments/1271398698596696117/1305792292338008084/IMG_20241112_161159_854.jpg?ex=67345156&is=6732ffd6&hm=6ef56215509bb035bdff2cf1b008da0b5f302b163483a4b7bb8b3ec8bf590fd9&=&format=webp&width=522&height=585" width="200" height="200"/>
	<img src="https://media.discordapp.net/attachments/1271398698596696117/1305792701567995945/KakaoTalk_20241112_161327174.jpg?ex=673451b8&is=67330038&hm=23dc35c56dc9b555b4f1b84d9aa45df4bc6f4a451583d80e43d77cc38d471008&=&format=webp&width=780&height=585" width="200" height="200"/>
	<img src="https://i.pinimg.com/236x/d6/4e/97/d64e9765deca662e8fa07d2cfdb67f7c.jpg" width="200" height="200"/>
	<img src="https://i.pinimg.com/236x/52/33/cf/5233cf1dfa7cb3ddeee3bb286c11f3f8.jpg" width="200" height="200"/>
	<img src="https://avatars.githubusercontent.com/u/131550706?v=4" width="200" height="200"/>
</p>

|  &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp; 🐶 박화랑  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐙고유림  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐻김문수  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |     &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 😺신원영  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;   |  &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp; 🐶오창준  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |
|------------------------------------------|--------------------------------------|------------------------------------------|-----------------------------------|------------------------------------------|
| **streamlit** | **README, 아키텍처** | **RAG, schema** | **RAG(prompt), vector DB** | **PDF 벡터화, README 초안 작성** | 

<br>
<br>

# 🤖 보험 약관 RAG Chatbot

## 📌 소개

GPT-4o-mini 기반 **보험 약관 RAG 챗봇**입니다. 보험사의 내부 가상 상담원이 특정 상품의 보험 약관을 쉽게 찾을 수 있도록 설계되었습니다. 복잡하고 일반적이지 않은 보험 약관의 내용을 벡터 DB 형태로 저장하여 LLM에서 효율적으로 검색할 수 있도록 하였습니다.

<br>
<br>

## 📌 동기

특정 보험 약관의 내용은 복잡하고 일반적인 정보가 아니기 때문에 기존 LLM에서 쉽게 찾아볼 수 없습니다. 이를 해결하기 위해 보험 약관을 벡터 DB로 저장하고, 내부 상담원이 쉽게 약관을 조회할 수 있는 시스템을 구축하였습니다.

<br>
<br>

## 📌 기능

- **약관이 필요한 특정 상황에 대한 내용 설명 제공**

<br>

### 예시 | 

**질문:** 캐롯사의 해외여행보험에서 보장하는 척추지압술이나 침술의 치료한도는 얼마입니까?

- **일반 LLM의 답변:** 척추지압술에 대한 자세한 치료한도는 관련 약관을 찾아보시길 바랍니다.
- **RAG을 활용한 LLM의 답변:** 척추지압술이나 침술 치료의 한도는 하나의 질병에 대하여 US $1.000입니다.

<br>
<br>

## 🛠️ 기술 스택

![Python Badge](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white)
![LangChain Badge](https://img.shields.io/badge/LangChain-000000?style=flat&logo=&logoColor=white)
![OpenAI Badge](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=OpenAI&logoColor=white)
![PyPDFLoader Badge](https://img.shields.io/badge/PyPDFLoader-FFD43B?style=flat&logo=&logoColor=white)
![Streamlit Badge](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

<br>
<br>

## 🛠️ 설치

- Python 설치
- `.env` 파일에 `OPENAI_API_KEY` 추가

<br>
<br>

## 📌 사용 방법

프로젝트 디렉토리를 기준으로 `./data` 폴더가 필요합니다. `./data`에 내부 데이터로 활용하고자 하는 PDF를 넣으시면 됩니다.

1. **벡터 DB 생성**

   ```bash
   python vector_db.py
   ```

   - **입력:** `data` 폴더에 있는 PDF 파일들
   - **출력:** `chroma_db` 폴더 (벡터 DB 포함)

2. **스트림릿 실행**

   ```bash
   streamlit run main.py
   ```

<br>
<br>

## 📌 구현 사항

### 1. PDF를 벡터 DB화하기

- **PyPDFLoader**를 통해 보험 약관 PDF 파일 파싱
- 문서를 일정한 청크로 분할
- **OpenAI의 text-embedding-3-small 모델**을 통해 임베딩
- **Chroma**를 통해 데이터 벡터화

<br>

### 2. RAG 시스템 구현

- **LangChain** 기반으로 벡터 DB를 가져와 RAG 시스템 구현
- 모델은 **OpenAI의 GPT-4o-mini** 사용
- **RunnableWithMessageHistory** 인스턴스를 사용해 대화 내용을 기억하도록 구현
- 문서 기반의 신뢰성 있는 답변을 제공하기 위해 **temperature**를 1보다 낮게 설정

<br>
<br>

## 🖥️ 할루시네이션 테스트

**1-1. test 1**

- GPT vs RAG

<img src="https://cdn.discordapp.com/attachments/1271398698596696117/1306144236906745877/image.png?ex=6735991d&is=6734479d&hm=b2cba6582cb2a79e27551c2438c6d7b0dc1e3ba5794a8d3850ec048f71bfee74&"/>

- 본문

<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793352817442876/rag-2.png?ex=67345253&is=673300d3&hm=6b895b2e28f360929ca6c81c52849072c35509903432b939df3a72ddb42428c0&=&format=webp&quality=lossless&width=1122&height=536"/> <br>

**1-2. 결과** <br>
- GPT 응답은 정확한 한도를 제시하지 않고 플랜에 따라서 다를 수 있다고 언급하며, 구체적인 정보를 제공하지 않음
- RAG의 응답은 구체적인 한도 $1,000로 명시하고 있어 이 응답이 본문과 일치함

<br>
<br>

**2-1. test 2**

- GPT vs RAG

<img src="https://media.discordapp.net/attachments/1271398698596696117/1306145622071578695/image.png?ex=67359a67&is=673448e7&hm=ead275443178ec58790e898748a53a86cd83ae367d4bd9d3870a47682080ffb5&=&format=webp&quality=lossless"/>

- 본문

<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793352569983006/rag-1.png?ex=67345253&is=673300d3&hm=0e47e4807f83e752f66a145c4c6b485061ac327c0a17c369c8eac96ba7ef0d91&=&format=webp&quality=lossless&width=1161&height=392"/> <br>

**2-2. 결과**<br>
- GPT의 답변은 실제 약관의 내용을 반영하지 않고, 일반적인 원칙에 대한 설명을 하고 있음
- 본문에서 우체국 소인이 찍힌 날로부터 3일이 지나면 회사에 접수된 것이라고 명시되어 있으며, RAG 답변 또한 동일하게 3일이라고 답변함
<br>
> RAG 기반 응답이 본문과 더 일치하며, GPT 응답은 상대적으로 모호한 정보를 제공함

<br>
<br>

## 🖥️ 프로젝트 진행에서의 문제 발생과 해결

### 1. 이전 대화 내용을 기억하지 못함

- **문제:** 이전 대화 내용을 기억하지 못하여 대화의 연속성이 떨어짐
- **해결:** `RunnableWithMessageHistory` 메모리를 사용하여 사용자와의 대화 맥락을 기억하도록 구현

### 2. 보험 약관 간 내용의 유사성으로 인한 벡터 DB 혼선

- **문제:** 다른 보험사의 동일한 보험 약관 간 내용의 유사성 때문에 벡터 DB에서 혼선 발생
- **해결:** 찾는 보험의 종류를 선택하게 하고, 해당 보험의 약관에서만 탐색하도록 구현

### 3. Streamlit에서 대화 UI 개선

- **문제:** Streamlit 구현 과정에서, OpenAI의 ChatGPT UI처럼 대화가 아래로 스크롤되기를 원함
- **해결:** `st.chat_message`를 통해 메시지를 순차적으로 표시되도록 하여 자동으로 위에서 아래로 스크롤 되도록 구현

### 4. Streamlit 에서 아바타와 유저의 아이콘
- 문제: 질문을 할 때 마다 유저와 아바타의 아이콘이 초기화되어 나타남
- 해결: 유저와 아바타의 아이콘을 with문을 통해 아이콘 지속

### 5. 할루시네이션 
- 문제: 할루시네이션 발생
- 해결: meta data를 통하여 출력에 제한을 적용했지만, 의도대로 답변하지 못하는 경우 발생

<br>
<br>

## ✏️ Model Architecture
- PDF → Chroma 벡터화 → Chroma → LangChai ↔ OpenAI

<br>

- 사용자 질문 → Streamlit → LangChain → OpenAI → LangChain → Streamlit → 사용자
<br>
<br>

<img src="https://cdn.discordapp.com/attachments/1271398698596696117/1306148317075607573/image.png?ex=67359ce9&is=67344b69&hm=9948c91cb42202fcc932572237ba3965db09ffb8359d2758255dd4bb804c6aae&">


# 보험 약관 RAG 챗봇

**3번째 프로젝트 3조**

## 팀원 소개


<p align="center">
	<img src="https://media.discordapp.net/attachments/1271398698596696117/1305792292338008084/IMG_20241112_161159_854.jpg?ex=67345156&is=6732ffd6&hm=6ef56215509bb035bdff2cf1b008da0b5f302b163483a4b7bb8b3ec8bf590fd9&=&format=webp&width=522&height=585" width="200" height="200"/>
	<img src="https://media.discordapp.net/attachments/1271398698596696117/1305792701567995945/KakaoTalk_20241112_161327174.jpg?ex=673451b8&is=67330038&hm=23dc35c56dc9b555b4f1b84d9aa45df4bc6f4a451583d80e43d77cc38d471008&=&format=webp&width=780&height=585" width="200" height="200"/>
	<img src="https://i.pinimg.com/236x/d6/4e/97/d64e9765deca662e8fa07d2cfdb67f7c.jpg" width="200" height="200"/>
	<img src="https://i.pinimg.com/236x/52/33/cf/5233cf1dfa7cb3ddeee3bb286c11f3f8.jpg" width="200" height="200"/>
	<img src="https://media.discordapp.net/attachments/1271398698596696117/1305681879441346570/1724243923846.jpg?ex=6733ea82&is=67329902&hm=70ae78eeed4153d5c46a97c4268e6de581a03e21f47c7f65d242fa61f5531a19&=&format=webp&width=562&height=585" width="200" height="200"/>
</p>

|  &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp; 🐶 박화랑  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐱고유림  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐹김문수  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |     &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐰신원영  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;   |  &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp; 🐶오창준  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |
|------------------------------------------|--------------------------------------|------------------------------------------|-----------------------------------|------------------------------------------|
| streamlit, chroma DB | README | RAG | RAG, vector DB | PDF 벡터화, README 초안 작성 | 

## 소개

GPT-4o-mini 기반 **보험 약관 RAG 챗봇**입니다. 보험사의 내부 가상 상담원이 특정 상품의 보험 약관을 쉽게 찾을 수 있도록 설계되었습니다. 복잡하고 일반적이지 않은 보험 약관의 내용을 벡터 DB 형태로 저장하여 LLM에서 효율적으로 검색할 수 있도록 하였습니다.

<br>
<br>

## 동기

특정 보험 약관의 내용은 복잡하고 일반적인 정보가 아니기 때문에 기존 LLM에서 쉽게 찾아볼 수 없습니다. 이를 해결하기 위해 보험 약관을 벡터 DB로 저장하고, 내부 상담원이 쉽게 약관을 조회할 수 있는 시스템을 구축하였습니다.

<br>
<br>

## 기능

- **약관이 필요한 특정 상황에 대한 내용 설명 제공**

<br>

### 예시 | 

**질문:** 캐롯사의 해외여행보험에서 보장하는 척추지압술이나 침술의 치료한도는 얼마입니까?

- **일반 LLM의 답변:** 척추지압술에 대한 자세한 치료한도는 관련 약관을 찾아보시길 바랍니다.
- **RAG을 활용한 LLM의 답변:** 척추지압술이나 침술 치료의 한도는 하나의 질병에 대하여 US $1.000입니다.

<br>
<br>

## 기술 스택

![Python Badge](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white)
![LangChain Badge](https://img.shields.io/badge/LangChain-000000?style=flat&logo=&logoColor=white)
![OpenAI Badge](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=OpenAI&logoColor=white)
![PyPDFLoader Badge](https://img.shields.io/badge/PyPDFLoader-FFD43B?style=flat&logo=&logoColor=white)
![Streamlit Badge](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

<br>
<br>

## 설치

- Python 설치
- `.env` 파일에 `OPENAI_API_KEY` 추가

<br>
<br>

## 사용 방법

프로젝트 디렉토리를 기준으로 `./data` 폴더가 필요합니다. `./data`에 내부 데이터로 활용하고자 하는 PDF를 넣으시면 됩니다.

1. **벡터 DB 생성**

   ```bash
   python pdf2vector.py
   ```

   - **입력:** `data` 폴더에 있는 PDF 파일들
   - **출력:** `chroma_db` 폴더 (벡터 DB 포함)

2. **스트림릿 실행**

   ```bash
   streamlit run main.py
   ```

<br>
<br>

## 구현 사항

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

## 할루시네이션 테스트

**1-1. test 1**

- GPT 답변

<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793351592706078/gpt2.png?ex=67345253&is=673300d3&hm=f11c8b4fb851819ca7e6728a4613c53855a7d67debcfaa93aa1da67893cc5183&=&format=webp&quality=lossless&width=586&height=585"/>

- RAG 답변

<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793351903219712/rag-2.png?ex=67345253&is=673300d3&hm=fdaeaa0648de27cef7f0dc2ce259e436894913bc23d83dbf4067815bae3d9903&=&format=webp&quality=lossless&width=1160&height=366"/>

- 본문
<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793352817442876/rag-2.png?ex=67345253&is=673300d3&hm=6b895b2e28f360929ca6c81c52849072c35509903432b939df3a72ddb42428c0&=&format=webp&quality=lossless&width=1122&height=536"/> <br>

**1-2. 결과** <br>
- GPT 응답은 정확한 한도를 제시하지 않고 플랜에 따라서 다를 수 있다고 언급하며, 구체적인 정보를 제공하지 않음
- RAG의 응답은 구체적인 한도 $1,000로 명시하고 있어 이 응답이 본문과 일치함

<br>
<br>

**2-1. test 2**

- GPT 답변
<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793351311949894/gpt-1.png?ex=67345253&is=673300d3&hm=4f92b3359a9e7d8fd7e954e36d5aeff3887b778da5dd3dd499c1a31b97aa3e1d&=&format=webp&quality=lossless&width=1031&height=398"/>

- RAG 답변

<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793352335097897/rag-1.png?ex=67345253&is=673300d3&hm=520539bad693528331d53dade4ea271d8bccdc731fc9f203ef7eec877bc11104&=&format=webp&quality=lossless&width=490&height=323"/>

- 본문

<img src="https://media.discordapp.net/attachments/1271398698596696117/1305793352569983006/rag-1.png?ex=67345253&is=673300d3&hm=0e47e4807f83e752f66a145c4c6b485061ac327c0a17c369c8eac96ba7ef0d91&=&format=webp&quality=lossless&width=1161&height=392"/> <br>

**2-2. 결과**<br>
- GPT의 답변은 실제 약관의 내용을 반영하지 않고, 일반적인 원칙에 대한 설명을 하고 있음
- 본문에서 우체국 소인이 찍힌 날로부터 3일이 지나면 회사에 접수된 것이라고 명시되어 있으며, RAG 답변 또한 동일하게 3일이라고 답변함
<br>
> RAG 기반 응답이 본문과 더 일치하며, GPT 응답은 상대적으로 모호한 정보를 제공함

<br>
<br>

## 프로젝트 진행에서의 문제 발생과 해결

### 1. 이전 대화 내용을 기억하지 못함

- **문제:** 이전 대화 내용을 기억하지 못해 대화의 연속성이 떨어짐
- **해결:** `RunnableWithMessageHistory` 메모리를 사용하여 사용자와의 대화 맥락을 기억하도록 구현

### 2. 보험 약관 간 내용의 유사성으로 인한 벡터 DB 혼선

- **문제:** 다른 보험사의 동일한 보험 약관 간 내용의 유사성 때문에 벡터 DB에서 혼선 발생
- **해결:** 찾는 보험의 종류를 선택하게 하고, 해당 보험의 약관에서만 탐색하도록 구현

### 3. Streamlit에서 대화 UI 개선

- **문제:** Streamlit 구현 과정에서, OpenAI의 ChatGPT UI처럼 대화가 아래로 스크롤되기를 원함
- **해결:** ```st.chat_message```를 통해 메시지를 순차적으로 표시되도록 하여 자동으로 위에서 아래로 스크롤 되도록 구현

### 4. Streamlit 에서 아바타와 유저의 아이콘
- 문제: 질문을 할 때 마다 유저와 아바타의 아이콘이 초기화되어 나타남
- 해결: 유저와 아바타의 아이콘을 with문을 통해 아이콘 지속

<br>
<br>

## Model Architecture


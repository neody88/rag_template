# test_rag

## 1. install
```
pip install -r requirements.txt
```

## 2. run vllm
* vllm으로 embedding model 서빙
### 2.1 download model
```
cd scripts
bash download.model.sh
```
### 2.2. vllm serve
```
cd scripts
bash run_vllm.granite-embedding-278m-multilingual.sh
```

### 2.3. test api
```
curl http://localhost:8889/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-embedding-278m-multilingual",
    "input": "안녕하세요",
    "encoding_format": "float"
  }'
```

## 3. test code
* src/test_call_embedding.py
  * vllm api 호출 방법 라이브러리 2가지 비교
    * 1) openai
    * 2) lanchain_openai
  * openai vs lanchain_openai 호출 속도 비교
    ```
    # test gpu: v100 (using vllm)
    # response time
    #  - test_openai: 400ms
    #  - test_lanchain: 30ms
    ```
* src/test_embedding_search.py
  * lanchain_openai의 FAISS 라이브러리 활용한 벡터 서치 예제
    ```
    # 0. init embedding api
    # 1. 텍스트 파일 로드
    # 2. 벡터스토어 생성
    # 3. 벡터 검색
    ```

## 4. 주의 사항
* 랭체인 라이브러리 사용시 한글 쿼리 깨짐 주의
  * openai 라이브러리 보다 lanchain_openai 라이브러리 사용하여 api 호출시 분석 속도가 빠르나, 한글 쿼리 깨짐에 주의해야한다.
  * 아래 코드에서 check_embedding_ctx_length=False, tiktoken_enabled=False 옵션 설정 없이 embedding 호출시 한글 깨짐
  * vllm serve시 "export VLLM_LOGGING_LEVEL=DEBUG" 옵션을 설정하여 해당 코드 실행후 서버 로그 보면 prompt 값 한글이 깨지는지 안깨지는지 확인 가능.
  ```
  def test_lanchain():
    # 랭체인 라이브러리 사용시 한글 쿼리 깨짐 주의
    # - 이슈:
    #   - vllm 서버에 쿼리 로그보면 쿼리가 깨짐
    #   - 하지만 보통 사람들 이거 모름, 왜냐면 쿼리 깨져도 임베딩 결과 나옴
    #   - 하지만 유사도 계산하다보면 엉뚱한 문장들이 가깝다고 나올때 이상함을 느낌
    # - 원인:
    #   - 랭체인 코드 안에 컨텍스트길이 짜를 때 토큰 노멀라이즈 실행됨, 이때 한글 캐릭터 깨져서 토크나이즈됨
    #   - OpenAIEmbeddings가 tiktoken으로 토큰 카운팅할 때 문제가 생김
    #     - tiktoken=True: 토큰 수 계산 / 최대 길이 초과 시 텍스트 자르기 / 특수 문자 처리
    # - 해결방안:
    #   - 아래와 같이 옵션 파라미터 기입
    #   - check_embedding_ctx_length=False
    #   - tiktoken_enabled=False


    from langchain_openai import OpenAIEmbeddings

    emb_model = OpenAIEmbeddings(
        model="granite-embedding-278m-multilingual",
        openai_api_base="http://localhost:8889/v1",
        openai_api_key="dummy",
        check_embedding_ctx_length=False, # 한글 쿼리 깨짐 방지
        tiktoken_enabled=False #한글 쿼리 깨짐 방지
        )
  ```

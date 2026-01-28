import time

def test_openai():
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8889/v1",
        api_key="dummy"
    )
    
    start = time.time()
    response = client.embeddings.create(
        model="granite-embedding-278m-multilingual",
        input="안녕하세요"
    )

    end = time.time()
    print(f"소요시간: {end - start:.2f}초")
    print("response:")
    print(response.data[0].embedding)


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
    
    start = time.time()
    response = emb_model.embed_query("안녕하세요")
    end = time.time()
    print(f"소요시간: {end - start:.2f}초")
    print("response:")
    print(response)



if __name__=="__main__":
    # test gpu: v100 (using vllm)
    # response time
    #  - test_openai: 400ms
    #  - test_lanchain: 30ms
    test_openai()
    test_lanchain()

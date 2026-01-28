import time

# structured output schema
schema = {
    "title": "IntentClassification",
    "description": "사용자 의도 분류 결과",
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["greeting", "question", "complaint", "request", "feedback", "other"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"}
    },
    "required": ["intent", "confidence", "reason"]
}

def test_openai():
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8889/v1",
        api_key="dummy"
    )
    
    start = time.time()
    response = client.chat.completions.create(
        model="granite-4.0-micro",
        messages=[
            {"role": "system", "content": "Classify the user's intent."},
            {"role": "user", "content": "하이 헬로 안녕?"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "intent_classification",
                "schema": schema
            }
        },
        temperature=0,
        #max_tokens=256
    )
    end = time.time()
    print(f"소요시간: {end - start:.2f}초")
    print("response:")
    print(response.choices[0].message.content)


def test_lanchain():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import convert_to_messages

    llm = ChatOpenAI(
        model="granite-4.0-micro",
        openai_api_base="http://localhost:8889/v1",
        openai_api_key="dummy",
        temperature=0,
    )

    structured_llm = llm.with_structured_output(schema)

    raw_messages=[
        {"role": "system", "content": "Classify the user's intent."},
        {"role": "user", "content": "하이 헬로 안녕?"}
    ]

    start = time.time()
    response = structured_llm.invoke(convert_to_messages(raw_messages))
    end = time.time()
    print(f"소요시간: {end - start:.2f}초")
    print("response:")
    print(response)


if __name__=="__main__":
    # test gpu: v100 (using vllm)
    # response time
    #  - test_openai: 3.87초
    #  - test_lanchain: 3.63초
    test_openai()
    test_lanchain()

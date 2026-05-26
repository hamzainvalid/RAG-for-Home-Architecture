import json
    embedding_function=get_embedding_model()
)


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.3
)


def build_query(scene_data):
    issues = ", ".join(scene_data.get("issues", []))
    action = scene_data["user_action"]["action"]

    object_types = [obj["type"] for obj in scene_data["objects"]]

    return f"""
    User action: {action}
    Issues: {issues}
    Objects: {object_types}
    """


async def analyze_scene(scene_data):
    query = build_query(scene_data)

    docs = vector_store.similarity_search(query, k=5)

    rag_context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "human",
            f"""
            RAG Context:
            {rag_context}

            Scene Data:
            {json.dumps(scene_data, indent=2)}

            Return JSON in this format:
            {{
                "suggestions": [
                    {{
                        "target_id": "",
                        "action": "",
                        "direction": "",
                        "distance": 0,
                        "reason": ""
                    }}
                ],
                "warnings": [],
                "scores": {{}}
            }}
            """
        )
    ])

    chain = prompt | llm

    response = await chain.ainvoke({})

    return response.content
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from config import (
    OPENAI_API_KEY,
    MODEL_NAME
)

from prompts import SYSTEM_PROMPT

from qdrant_store import vector_store


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.3
)


def build_query(scene_data):

    issues = ", ".join(
        scene_data.get("issues", [])
    )

    scores = scene_data.get("scores", {})

    object_types = [
        obj["type"]
        for obj in scene_data["objects"]
    ]

    room_types = [
        room["room_type"]
        for room in scene_data["rooms"]
    ]

    return f"""
    User Action:
    {scene_data['user_action']['action']}

    Target:
    {scene_data['user_action']['target_id']}

    Issues:
    {issues}

    Scores:
    {json.dumps(scores)}

    Room Types:
    {room_types}

    Object Types:
    {object_types}
    """


async def analyze_scene(scene_data):

    query = build_query(scene_data)

    docs = vector_store.similarity_search(
        query=query,
        k=5
    )

    rag_context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_PROMPT
        ),
        (
            "human",
            f"""
            Retrieved Context:
            {rag_context}

            Scene Metadata:
            {json.dumps(scene_data, indent=2)}

            Return JSON ONLY.

            Required format:

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

              "scores": {{
                "spacing": 0,
                "walkability": 0,
                "style_consistency": 0
              }}
            }}
            """
        )
    ])

    chain = prompt | llm

    response = await chain.ainvoke({})

    return response.content
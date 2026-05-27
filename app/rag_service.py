import json

from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from .config import (
    OPENAI_API_KEY,
    MODEL_NAME
)

from .prompts import SYSTEM_PROMPT

from .qdrant_store import vector_store


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
    ("system", SYSTEM_PROMPT),
    ("human", """
        Retrieved Context:
        {rag_context}
        
        Scene Metadata:
        {scene_json}
        
        Return JSON ONLY.
        
        Required format:
        
        Return ONLY valid JSON with:
        - suggestions
        - warnings
        - scores
        
        Each suggestion must contain:
        - target_id
        - action
        - direction
        - distance
        - reason
        
        Each score must contain:
        - spacing
        - walkability
        - style_consistency
        """)
])

    chain = prompt | llm

    response = await chain.ainvoke({
        "rag_context": rag_context,
        "scene_json": json.dumps(scene_data, indent=2)
    })

    return response.content
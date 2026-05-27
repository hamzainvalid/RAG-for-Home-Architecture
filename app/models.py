from typing import Optional, List, Dict

from pydantic import BaseModel

class Vector3(BaseModel):
    x: float
    y: float
    z: float


class Dimensions(BaseModel):
    width: float
    height: float
    depth: float

class Clearance(BaseModel):
    front: float
    back: float
    left: float
    right: float

class Relationships(BaseModel):
    attached_to: Optional[str] = None
    faces: Optional[str] = None
    near: Optional[List[str]] = []


class ObjectState(BaseModel):
    selected: bool
    movable: bool
    resizable: bool


class SceneObject(BaseModel):
    id: str
    type: str
    category: str
    subtype: Optional[str] = None
    room_id: Optional[str] = None

    position: Vector3
    rotation: Vector3
    scale: Vector3

    dimensions: Dimensions

    style: List[str] = []
    material: List[str] = []
    color: List[str] = []
    tags: List[str] = []
    constraints: List[str] = []

    relationships: Relationships
    clearance: Clearance

    walkable_impact: Optional[float] = 0
    importance_score: Optional[float] = 0

    state: ObjectState


class Room(BaseModel):
    room_id: str
    room_type: str
    style: List[str] = []
    objects: List[str] = []


class UserAction(BaseModel):
    action: str
    target_id: str


class SceneData(BaseModel):
    scene_id: str
    user_action: UserAction
    rooms: List[Room]
    objects: List[SceneObject]
    scores: Dict[str, float]
    issues: List[str]


class SceneRequest(BaseModel):
    scene_data: SceneData
from sqlalchemy import Column, Integer, String, LargeBinary
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    name = Column(String)
    face_front_embedding = Column(LargeBinary)
    face_left_embedding = Column(LargeBinary)
    face_right_embedding = Column(LargeBinary)

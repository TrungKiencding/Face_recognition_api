from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from sqlalchemy.orm import Session
import numpy as np
import cv2
from database import get_db
from services.face_recognition_service import FaceRecognitionService
from typing import List

router = APIRouter(
    prefix="/face",
    tags=["face-recognition"],
    responses={404: {"description": "Not found"}},
)

face_service = FaceRecognitionService()

@router.post("/register")
async def register_face(id: str = Form(...), name: str = Form(...), image_front: UploadFile = File(...), image_left: UploadFile = File(...), image_right: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Đăng ký khuôn mặt mới vào hệ thống
    """
    def read_and_decode_image(image: UploadFile):
        contents = image.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    try:
        image_front = read_and_decode_image(image_front)
        image_left = read_and_decode_image(image_left)
        image_right = read_and_decode_image(image_right)
    
        # Đăng ký khuôn mặt
        result = face_service.register_face(db, id, name, image_front, image_left, image_right)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize")
async def recognize_face(image: UploadFile = File(...), db: Session = Depends(get_db), id: str = Form(...)):
    """
    Nhận diện khuôn mặt từ ảnh
    """
    try:
        # Đọc và chuyển đổi ảnh
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Nhận diện khuôn mặt
        result = face_service.recognize_face(db, img, id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["message"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
def get_all_users(db: Session = Depends(get_db)):
    """
    Lấy danh sách tất cả người dùng
    """
    return face_service.get_all_users(db)

@router.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Xóa người dùng khỏi hệ thống
    """
    result = face_service.delete_user(db, user_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

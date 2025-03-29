import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from sqlalchemy.orm import Session
import models
import pickle
from typing import List, Optional
import tensorflow as tf

class FaceRecognitionService:
    def __init__(self):
        # Khởi tạo MTCNN detector
        self.detector = MTCNN()
        
        # Tải model FaceNet
        self.interpreter = tf.lite.Interpreter(model_path='model/fast_facenet.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        #Ngưỡng khoảng cách để xác định khuôn mặt
        self.threshold = 0.35
    
    def detect_face(self, image):
        """Phát hiện khuôn mặt trong ảnh sử dụng MTCNN"""
        # Chuyển đổi ảnh sang định dạng RGB nếu cần
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Phát hiện khuôn mặt
        faces = self.detector.detect_faces(image)
        
        if not faces:
            return None
        
        # Lấy khuôn mặt có độ tin cậy cao nhất
        face = max(faces, key=lambda x: x['confidence'])
        
        # Trích xuất bounding box
        x, y, width, height = face['box']
        
        # Đảm bảo tọa độ không âm
        x, y = max(0, x), max(0, y)
        
        # Cắt khuôn mặt từ ảnh
        face_img = image[y:y+height, x:x+width]
        
        # Điều chỉnh kích thước ảnh cho phù hợp với đầu vào của FaceNet
        face_img = cv2.resize(face_img, (112, 112))
         
        return face_img
    
    def get_embedding(self, face_img):
        """Trích xuất embedding từ ảnh khuôn mặt"""
        # Tiền xử lý ảnh
        face_img = face_img.astype('float32')
        mean, std = face_img.mean(), face_img.std()
        face_img = (face_img - mean) / std
        
        # Mở rộng kích thước để phù hợp với đầu vào của model
        face_img = np.expand_dims(face_img, axis=0)
        
        # Đặt input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], face_img)
        
        # Thực hiện inference
        self.interpreter.invoke()
        
        # Lấy kết quả từ output tensor
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        return embedding
    
    def register_face(self, db: Session, id: str, name: str, image_front, image_left, image_right):
        """Đăng ký khuôn mặt mới vào database"""
        # Phát hiện khuôn mặt
        face_img_front = self.detect_face(image_front)
        face_img_left = self.detect_face(image_left)
        face_img_right = self.detect_face(image_right)
        
        if face_img_front is None:
            return {"success": False, "message": "Không phát hiện khuôn mặt trong ảnh mặt trước"}
        if face_img_left is None:
            return {"success": False, "message": "Không phát hiện khuôn mặt trong ảnh bên trái"}
        if face_img_right is None:
            return {"success": False, "message": "Không phát hiện khuôn mặt trong ảnh bên phải"}
        
        # Trích xuất embedding
        embedding_front = self.get_embedding(face_img_front)
        embedding_left = self.get_embedding(face_img_left)
        embedding_right = self.get_embedding(face_img_right)
        
        # Kiểm tra xem người dùng đã tồn tại chưa
        existing_user = db.query(models.User).filter(models.User.id == id).first()
        if existing_user:
            return {"success": False, "message": f"ID {id} đã tồn tại"}
        
        # Lưu embedding vào database
        db_user = models.User(id = id, name=name, face_front_embedding=pickle.dumps(embedding_front), face_left_embedding=pickle.dumps(embedding_left), face_right_embedding=pickle.dumps(embedding_right))
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return {"success": True, "message": f"Đã đăng ký thành công người dùng {name}", "user_id": db_user.id}
    
    def recognize_face(self, db: Session, image, id: str):
        """Nhận diện khuôn mặt từ ảnh"""
        # Phát hiện khuôn mặt
        face_img = self.detect_face(image)
        
        if face_img is None:
            return {
                "success": True, 
                "message": "Không phát hiện khuôn mặt trong ảnh"}
        
        # Trích xuất embedding
        embedding = self.get_embedding(face_img)
        
        # Lấy tất cả người dùng từ database
        user = db.query(models.User).filter(models.User.id == id).first()
        
        if not user:
            return {
                "success": True, 
                "message": "Không tìm thấy người dùng với ID {id}"}
        
        # Tìm người dùng có khoảng cách embedding nhỏ nhất
        user_embeddings = [
            pickle.loads(user.face_front_embedding),
            pickle.loads(user.face_left_embedding),
            pickle.loads(user.face_right_embedding)
        ]
        
        min_distance = float('inf')
        for user_embedding in user_embeddings:
            # Tính cosine similarity giữa các embedding
            cosine_similarity = np.dot(embedding, user_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(user_embedding))
            distance = 1 - cosine_similarity
            min_distance = min(distance, min_distance)
            
        if min_distance <= self.threshold:
            return {
                "success": True, 
                "message": f"Nhận diện thành công", 
                "user_id": user.id,
                "name": user.name,
                "confidence": float(1 - min_distance) # Chuyển đổi khoảng cách thành độ tin cậy
            }
        else:
            return {"success": True, 
                    "message": "Khuôn mặt không khớp",
                    "confidence": float(1 - min_distance),
                    "name": user.name}
    
    def get_all_users(self, db: Session):
        """Lấy tất cả người dùng từ database"""
        users = db.query(models.User).all()
        return [{"id": user.id, "name": user.name} for user in users]
    
    def delete_user(self, db: Session, user_id: int):
        """Xóa người dùng khỏi database"""
        user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if not user:
            return {"success": False, "message": f"Không tìm thấy người dùng với ID {user_id}"}
        
        db.delete(user)
        db.commit()
        
        return {"success": True, "message": f"Đã xóa người dùng {user.name}"}

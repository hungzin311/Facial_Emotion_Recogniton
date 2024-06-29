import torch
from torchvision import transforms
from model import ResNet 
from model2 import MobileNet
import face_recognition
import numpy as np
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QMenu, QWidget
from PyQt6.QtGui import QPixmap, QAction, QPainter, QFont, QColor
from PyQt6.QtCore import Qt
import cv2

#Choose the model to use (resnet50 with higher accuracy but delay more time) (recommend to use mobilenet_v2 in real time problem)
model_path = 'model\\mobilenet_v2_final.pth'

#The model_path and model needed to be appropriate (Resnet50 with 'resnet50_final.pth' and MobileNetV2 with 'mobilenet_v2_final.pth')
#You need to choose the model_path appropriate with your chosen model
model = MobileNet()

#Select device
model.load_state_dict(torch.load(model_path, map_location= 'cpu'))
device = torch.device('cpu')

#The label of emotions
emotion  = ['suprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

#preprocess image
eval_transform  = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225]),
])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentimental Facial Expression Analyzation")
        self.setGeometry(350, 100, 900, 500)  # Set kích thước và vị trí cửa sổ

        #Tạo MenuBar
        select_image_menu = QMenu("Select image",self)
        self.menuBar().addMenu(select_image_menu)
        select_from_computer = QAction("Browse the computer",self)
        select_from_computer.triggered.connect(self.open_file_dialog)
        select_from_camera = QAction("Use camera",self)
        select_from_camera.triggered.connect(self.capture_image)
        
        
        select_image_menu.addAction(select_from_computer)
        select_image_menu.addAction(select_from_camera)
        
        
        #Tạo nơi hiển thị ảnh
        middle_frame = QHBoxLayout()

        # Khung hiển thị ảnh trái
        self.left_image_label = QLabel()
        self.left_image_label.setStyleSheet("border: 1px solid black;")

        # Khung hiển thị ảnh phải
        self.right_image_label = QLabel()
        self.right_image_label.setStyleSheet("border: 1px solid black;")

        middle_frame.addWidget(self.left_image_label)
        middle_frame.addWidget(self.right_image_label)
        
        main_layout = QVBoxLayout()
        
        analyze_image_button = QPushButton("Phân tích Ảnh")
        analyze_image_button.clicked.connect(self.analyze_image)
        
        
        main_layout.addLayout(middle_frame)
        main_layout.addWidget(analyze_image_button)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        
    def analyze_image(self):
        #PUSH CAI HAM XU LY ANH VAO DAY
        #self.path_to_picture là cái ảnh nhé 
        #phân tích xong thì m cho đường dẫn của ảnh đã được xử lý vào câu lệnh dưới 
        #self.load_image("{đường dẫn}", self.right_image_label)
        image_path = self.path_to_picture
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        face_image = image 
        for face_location in face_locations:
            top, right, bottom, left = face_location
            temp  = image[top:bottom, left:right]
            face_image = temp
        if(self.ok == 1):
            face_image = cv2.imread(image_path)
        if(self.ok == 0): 
            face_image = face_image
        face_image = face_image[:, :, ::-1]
        face_image = eval_transform(face_image)
        face_image= torch.unsqueeze(face_image, dim = 0)
        with torch.no_grad():
            model.eval()
            face_image = face_image.to(device)
            outputs, _ = model(face_image)
            _, predicts = torch.max(outputs, 1)
            x = emotion[predicts]
        # Tạo một QLabel để hiển thị ảnh
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(400,400,Qt.AspectRatioMode.KeepAspectRatio)
        image_label.setPixmap(pixmap)

        # Tạo một QPainter để vẽ lên ảnh
        painter = QPainter(pixmap)
        # Tạo một font
        font = QFont()
        font.setBold(True)
        font.setPointSize(20)
        painter.setFont(font)
        # Chọn màu nền xanh
        painter.setBrush(QColor(108, 123, 139))
        # Vẽ một hình chữ nhật với kích thước phù hợp để chứa văn bản
        # painter.drawRect(0, 0, pixmap.width(), 40)
        # Đặt màu chữ là trắng
        painter.setPen(QColor(255, 250, 240))
        text_width = painter.fontMetrics().boundingRect(x).width()
        text_height = painter.fontMetrics().height()
        text_x = (int)((pixmap.width() - text_width) / 2)
        text_y = (int)((pixmap.height() - text_height) / 2)

        painter.drawRect(0, 0, pixmap.width(), 30)
        painter.drawText(text_x, 25, x)
        # Vẽ văn bản
        # Kết thúc QPainter
        painter.end()
        
        self.load_image(pixmap, self.right_image_label)
    
        
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.path_to_picture = file_path
        self.ok = 1
        if file_path:
            pixmap = QPixmap(file_path)
            self.load_image(pixmap, self.left_image_label)
            

    def capture_image(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Không thể mở camera. Kiểm tra lại index hoặc camera có được kết nối không.")
            exit()
                # Đợi cho đến khi có frame từ camera
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể nhận frame từ camera.")
                break
            frame = cv2.flip(frame, 1)

            # Hiển thị frame
            image = np.array(frame)
            face_locations = face_recognition.face_locations(image)
            
            face_image = image
            for face_location in face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, bottom), (right, top-20), (0, 255, 0), 2)
                temp  = image[top-20:bottom, left:right]
                face_image = temp
                face_image = face_image[:, :, ::-1]
                face_image = eval_transform(face_image)
                face_image= torch.unsqueeze(face_image, dim = 0)
                with torch.no_grad():
                    model.eval()
                    face_image = face_image.to(device)
                    outputs, _ = model(face_image)
                    _, predicts = torch.max(outputs, 1)
                    x = emotion[predicts]
                cv2.putText(frame, x, (left, top -30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            
            
            cv2.imshow('camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite("captured_image.jpg", frame) 
                self.path_to_picture = "captured_image.jpg"
                self.ok = 0
                break

        cap.release()
        cv2.destroyAllWindows()
        self.load_image(QPixmap("captured_image.jpg"), self.left_image_label)
        

    def load_image(self, pixmap, label):
        try:
            # Lấy kích thước của QLabel
            label_width = label.width()
            label_height = label.height()
            
            # Thay đổi kích thước của ảnh sao cho vừa với QLabel
            pixmap = pixmap.scaled(label_width, label_height, Qt.AspectRatioMode.KeepAspectRatio)
            
            # Đặt ảnh đã thay đổi kích thước vào QLabel
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 0px solid black;")
        except Exception as e:
            print(f"Error loading image: {e}")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
import sys
import torch
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PIL import Image
from torchvision import models
import torch.nn as nn
from PyQt5.QtGui import QPixmap

# Define your custom model class
class RPSModel(nn.Module): #defining your custom model class
    def __init__(self, n_filters):
        super().__init__()

        self.n_filters = n_filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_filters,kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)        
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=self.n_filters, kernel_size=3)        
        
        self.flatten = nn.Flatten()    
        self.linear1 = nn.Linear(in_features=self.n_filters*5*5, out_features=50)
        self.linear2 = nn.Linear(50, 3)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X):
        #Featurizer
        x = self.conv1(X)
        x = self.relu(x)
        x= self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #Classifier
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)        
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.model = None
        self.loaded_image = None

    def initUI(self):
        self.setGeometry(100, 100, 500, 400)
        self.setWindowTitle("Image Inference App")

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        self.inference_button = QPushButton("Run Inference", self)
        self.inference_button.clicked.connect(self.run_inference)
        layout.addWidget(self.inference_button)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.result_label = QLabel("Predicted Class:", self)
        layout.addWidget(self.result_label)

        self.result_textbox = QTextEdit(self)
        self.result_textbox.setReadOnly(True)
        layout.addWidget(self.result_textbox)

        container = QWidget(self)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)

        if file_name:
            self.loaded_image = Image.open(file_name)
            self.result_textbox.clear()
            self.result_label.setText("Predicted Class:")
            self.result_textbox.setText("Image loaded.")

            # pixmap = QPixmap(file_name)
            # self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=True))

            pixmap = QPixmap(file_name)
            resized_pixmap = pixmap.scaled(500, 500, aspectRatioMode=True)
            self.image_label.setPixmap(resized_pixmap)


    def run_inference(self):
        if self.loaded_image is not None:
            if self.model is None:
                # self.model = RPSModel(n_filters=16)  # You can adjust the number of filters as needed
                # self.model.load_state_dict(torch.load("best_rps_200.pth", map_location=torch.device('cpu')))
                
                checkpoint = torch.load("best_rps_200.pth", map_location=torch.device('cpu'))
                # Create an instance of the model class
                self.model = RPSModel(n_filters=16)
                # Restore state for model and optimizer
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])

            input_image = transform(self.loaded_image).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_image)
                _, predicted_class = torch.max(output, 1)

            class_labels = ["Rock", "Paper", "Scissors"]  # Replace with your actual class labels
            predicted_label = class_labels[predicted_class]

            self.result_label.setText("Predicted Class:")
            self.result_textbox.setText(predicted_label)

        else:
            self.result_label.setText("Predicted Class:")
            self.result_textbox.setText("Load an image first.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())

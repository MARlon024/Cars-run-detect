import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
from easyocr import Reader
import numpy as np

# Define the license plate detection model
class LicensePlateDetector(nn.Module):
    def __init__(self, num_classes):
        super(LicensePlateDetector, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 36
model = LicensePlateDetector(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Initialize EasyOCR reader
reader = Reader(['en'], gpu=False)  # Set the languages you want to recognize

# Define the transformations for preprocessing the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the video
video_path = 'TEST.mp4'
cap = cv2.VideoCapture(video_path)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_np = np.array(image_pil)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # Perform license plate detection
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_label = torch.round(torch.sigmoid(outputs))
        predicted_label = predicted_label.squeeze()  # Remove the batch dimension

# Check if any class is predicted
    if torch.any(predicted_label == 1):
        result = reader.readtext(image_np)
        if result:
            plate_text = result[0][1]
            cv2.putText(frame, plate_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('License Plate Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()

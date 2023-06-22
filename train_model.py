import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
import xmltodict
from sklearn.metrics import accuracy_score
# Define the license plate recognition model
class LicensePlateRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(LicensePlateRecognizer, self).__init__()
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

# Define the custom license plate dataset
# Define the custom license plate dataset
class LicensePlateDataset(Dataset):
    def __init__(self, root, transform=None, resize_size=(224, 224)):
        self.root = root
        self.transform = transform
        self.resize_size = resize_size
        self.image_files = sorted(os.listdir(os.path.join(root, 'images')))
        self.annotation_files = sorted(os.listdir(os.path.join(root, 'annotations')))
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root, 'images', self.image_files[index])
        annotation_path = os.path.join(self.root, 'annotations', self.annotation_files[index])
        
        image = self.load_image(image_path)
        target = self.parse_annotation(annotation_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    def __len__(self):
        return len(self.image_files)
    
    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        return image
    
    def parse_annotation(self, path):
        with open(path, 'r') as f:
            annotation = xmltodict.parse(f.read())
        
        targets = []
        objects = annotation['annotation']['object']
        
        if isinstance(objects, list):
            for obj in objects:
                bbox = obj['bndbox']
                xmin = int(bbox['xmin'])
                ymin = int(bbox['ymin'])
                xmax = int(bbox['xmax'])
                ymax = int(bbox['ymax'])
                targets.append([xmin, ymin, xmax, ymax])
        elif isinstance(objects, dict):
            bbox = objects['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])
            targets.append([xmin, ymin, xmax, ymax])
        
        if len(targets) == 0:
            targets = torch.zeros((36,), dtype=torch.float32)  # Assign class probability 0 for no license plate
        else:
            targets = torch.ones((36,), dtype=torch.float32)  # Assign class probability 1 for license plate
        
        return targets



# Rest of the code remains the same

# Define the dataset and data loader
resize_size = (224, 224)  # Specify the desired size for resizing
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor()
])

dataset = LicensePlateDataset(root='archive', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create an instance of the license plate recognition model
num_classes = 36  # Assuming you are detecting license plates as a single class
model = LicensePlateRecognizer(num_classes=num_classes)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Using Binary Cross Entropy with Logits Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

NUM_EPOCHS = 25

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    true_labels = []
    predicted_labels = []

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Collect true and predicted labels for accuracy calculation
        true_labels.extend(targets.cpu().numpy().astype(int))
        predicted_labels.extend((outputs.sigmoid().detach().cpu().numpy() > 0.5).astype(int))

    
    epoch_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss}, Accuracy: {accuracy}")


# Save the trained model
torch.save(model.state_dict(), 'model.pt')

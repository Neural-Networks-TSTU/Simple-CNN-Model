import torch
from PIL import Image
from torchvision import transforms
from models.cnn_model import CNNClassifier
from models.lite_cnn_model import LightCNNClassifier
import argparse

THRESHOLD = 0.6
INPUT_SIZE = 96

def load_model(path, device):
    chkpt = torch.load(path, map_location=device)
    model = CNNClassifier()
    model.load_state_dict(chkpt['model_state'])
    model.to(device).eval()
    return model, chkpt['classes']

def preprocess(img_path):
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path)
    return tf(img).unsqueeze(0)  # (1,1,96,96)

def predict(model, tensor, device, classes):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        p, idx = torch.max(probs, 0)
        if p.item() < THRESHOLD:
            return "unknown", p.item()
        return classes[idx.item()], p.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="1.png")
    parser.add_argument("--model", default="checkpoints/best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, classes = load_model(args.model, device)

    img_t = preprocess(args.image)
    label, prob = predict(net, img_t, device, classes)
    print(f"Prediction: {label} (conf={prob:.3f})")

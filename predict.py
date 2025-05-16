import torch
from PIL import Image
from torchvision import transforms
from cnn_model import CNNClassifier
import argparse

CLASSES = ['ACB_1', 'ACB_2', 'ACB_MO_1', 'ACB_PC_1',
           'KM_1', 'SW_1', 'SW_2']
THRESHOLD = 0.6
INPUT_SIZE = 96

def load_model(path, device):
    model = CNNClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def preprocess(img_path):
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path)
    return tf(img).unsqueeze(0)  # (1,1,H,W)

def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        p, idx = torch.max(probs, 0)
        if p.item() < THRESHOLD:
            return "unknown", p.item()
        return CLASSES[idx.item()], p.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to png/jpg image", default="data/ACB_1/ACB_1_00005.jpg")
    parser.add_argument("--model", default="checkpoints/best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args.model, device)
    img_t = preprocess(args.image)
    label, prob = predict(net, img_t, device)
    print(f"Prediction: {label} (conf={prob:.3f})")

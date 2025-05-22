import torch
from PIL import Image
from torchvision import transforms, models
import argparse

THRESHOLD = 0.6
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, device, model_type):
    chkpt = torch.load(path, map_location=device)
    classes = chkpt['classes']

    if model_type == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    elif model_type == 'cnn':
        from models.cnn_model import CNNClassifier
        model = CNNClassifier()
    elif model_type == 'lightcnn':
        from models.lite_cnn_model import LightCNNClassifier
        model = LightCNNClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(chkpt['model_state'])
    model.to(device).eval()
    return model, classes

def preprocess(img_path, input_size=INPUT_SIZE):
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = Image.open(img_path)
    return tf(img).unsqueeze(0)

def predict(model, tensor, device, classes, threshold=THRESHOLD):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        p, idx = torch.max(probs, 0)
        if p.item() < threshold:
            return "unknown", p.item()
        return classes[idx.item()], p.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for image classification")
    parser.add_argument("--image", default="1.png")
    parser.add_argument("--model", default="checkpoints/best.pth")
    parser.add_argument("--model-type", choices=['lightcnn', 'cnn', 'resnet18'], default='resnet18')
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    net, classes = load_model(args.model, DEVICE, args.model_type)

    img_t = preprocess(args.image, input_size=INPUT_SIZE if args.model_type != 'resnet18' else 224)

    label, prob = predict(net, img_t, DEVICE, classes, threshold=args.threshold)
    print(f"Prediction: {label} (conf={prob:.3f})")
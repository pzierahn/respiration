import torch
import torchvision.transforms as transforms


def preprocess_frames(frames, size=(128, 128)):
    # Preprocess the frames to be in 128x128 with torch
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    # Transform each frame
    transformed_frames = torch.stack([
        transform(frame) for frame in frames
    ])

    return transformed_frames.unsqueeze(0)

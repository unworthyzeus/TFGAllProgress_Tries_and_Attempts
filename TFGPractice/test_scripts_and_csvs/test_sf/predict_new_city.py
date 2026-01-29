import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your trained model file
MODEL_PATH = "ckm_multicity_model.pth" 

# Path to the NEW city map you want to process
# (Change this to your new file)
INPUT_MAP_PATH = "C:\\Users\\guill\\OneDrive\\Documents\\TFG\\TFGPractice\\test\\San Fransisco\\San Fransisco.jpg" 

# Where to save the results
OUTPUT_DIR = "predicted_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The names of the 5 output channels (in the order trained)
CHANNEL_NAMES = ["Channel_Gain", "AoA_Phi", "AoA_Theta", "AoD_Phi", "AoD_Theta"]

# ==========================================
# MODEL DEFINITION (Must match training!)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5); x = torch.cat([x4, x], dim=1); x = self.conv1(x)
        x = self.up2(x); x = torch.cat([x3, x], dim=1); x = self.conv2(x)
        x = self.up3(x); x = torch.cat([x2, x], dim=1); x = self.conv3(x)
        x = self.up4(x); x = torch.cat([x1, x], dim=1); x = self.conv4(x)
        return self.outc(x)

# ==========================================
# PREDICTION LOGIC
# ==========================================
def predict():
    # 1. Setup
    if not os.path.exists(INPUT_MAP_PATH):
        print(f"Error: Input map '{INPUT_MAP_PATH}' not found.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(n_channels=1, n_classes=5).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    model.eval() # Switch to evaluation mode (turns off BatchNorm/Dropout randomness)

    # 3. Process Input Image
    print("Processing input map...")
    input_img = Image.open(INPUT_MAP_PATH).convert('L') # Force grayscale
    
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # Normalizes to [0, 1]
    ])
    
    input_tensor = preprocess(input_img).unsqueeze(0).to(DEVICE) # Add batch dimension -> [1, 1, 128, 128]

    # 4. Run Inference
    with torch.no_grad(): # Disable gradient calculation for speed
        output_tensor = model(input_tensor)

    # 5. Save & Visualize Results
    print("Saving results...")
    
    # Remove batch dimension -> [5, 128, 128]
    output_tensor = output_tensor.squeeze(0).cpu() 
    
    plt.figure(figsize=(15, 8))
    
    # Plot Input
    plt.subplot(2, 3, 1)
    plt.title("Input City Map")
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')

    # Plot Outputs
    for i, name in enumerate(CHANNEL_NAMES):
        channel_data = output_tensor[i].numpy()
        
        # Save as individual PNG
        save_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        plt.imsave(save_path, channel_data, cmap='inferno') # 'inferno' is good for heatmaps
        
        # Add to plot
        plt.subplot(2, 3, i+2)
        plt.title(f"Pred: {name}")
        plt.imshow(channel_data, cmap='inferno')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "full_comparison.png"))
    print(f"Done! Check the '{OUTPUT_DIR}' folder.")
    plt.show()

if __name__ == "__main__":
    predict()
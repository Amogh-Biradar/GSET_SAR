import torch

# Fix denoiser_model.pth
try:
    model1 = torch.load("denoiser_model.pth", map_location="cpu", encoding="latin1")
    torch.save(model1, "denoiser2.pth")
    print("✅ Saved denoiser2.pth in binary mode.")
except Exception as e:
    print("❌ Failed to convert denoiser_model.pth:", e)

# Fix scream_classifier.pth
try:
    model2 = torch.load("scream_classifier.pth", map_location="cpu", encoding="latin1")
    torch.save(model2, "scream2.pth")
    print("✅ Saved scream2.pth in binary mode.")
except Exception as e:
    print("❌ Failed to convert scream_classifier.pth:", e)

# rps_inference

In this code I use pyqt to build a simple User Interface for Inference

- We use Rock Scissors Paper dataset to train the model before hand
- After training we use this inference code to load image and display it in UI
- Then we call inference on the loaded image and display its predicted label in the UI

Replace the model name in torch.load('best_rps_200.pth') command with your own rps model

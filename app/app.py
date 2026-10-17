"""Live demo (Gradio): draw a digit and compare predictions from every model.

TODO:
- Sketchpad input for drawing a digit.
- Dropdown to pick a model, or show all models side by side.
- Preprocess the drawing with src/preprocess.py.
- Show the predicted digit and the probability bar chart (like predict_single in the notebook).
- Show the preprocessed 28x28 image so users can see what the model actually receives.
- Load models from models/ only once, at startup.
- Deploy to Hugging Face Spaces (it runs app.py with app/requirements.txt).
"""

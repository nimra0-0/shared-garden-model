from fastai.vision.all import *
import gradio as gr

#load the model
learn = load_learner('flower_classifier.pkl')
labels = learn.dls.vocab


def predict(img):
    """
    Predict the class of the image
    Args:
        img: PIL.Image.Image
    Returns:
    dict: A dictionary of the predicted class and the probability
    """
    
    if img is None:
        return {label: 0.0 for label in labels}

    if isinstance(img, dict) and "composite" in img: # For Doodle Sketch we are interested in the composite image
        img = img["composite"]  # Extract PIL.Image from Sketchpad dict
    
    # --- Convert NumPy array to PIL Image ---
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # --- Convert RGBA -> RGB (or grayscale 'L' for B/W) ---
    if img.mode == "RGBA":
        img = img.convert("L")

    fastai_img = PILImage.create(img) #The pkl model expects a fastai image

    pred, pred_idx, probs = learn.predict(fastai_img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

with gr.Blocks() as demo:
    gr.Markdown("## Flower Classifier: Upload an Image or Doodle a Sketch")

    #Create Tabs: one for uploading images, one for sketching doodles
    with gr.Tab("Upload Image"):
        img_upload = gr.Image(
            type="pil",
             label="Upload an image",
             height=400
             )
        img_upload_button = gr.Button("Upload")
    with gr.Tab("Sketch Doodle"):
        doodle_sketch = gr.Sketchpad(
                label="Draw a flower doodle",
                brush=gr.Brush(
                            colors=["#000000"],  # Default color (black)
                            default_size=4,      # Default brush size
                        )
            )
        doodle_sketch_button = gr.Button("Submit Sketch")
    
    #output the predicted class and the probability
    output = gr.Label(num_top_classes=2, label="Predicted Class")

    #when the user uploads an image or sketches a doodle, predict the class
    img_upload_button.click(predict, inputs=img_upload, outputs=output)
    doodle_sketch_button.click(predict, inputs=doodle_sketch, outputs=output)

if __name__ == "__main__":
    demo.launch()
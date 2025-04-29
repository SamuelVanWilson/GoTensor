import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# 1. Load pre-trained style transfer model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# 2. Helper to load and preprocess images
def load_image(image_path, image_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img = np.array(img) / 255.0  # normalize to [0,1]
    img = img[np.newaxis, ...]   # add batch dimension
    return img.astype(np.float32)

# 3. Paths to your content and style images
content_path = 'content.jpg'
style_path = 'style.jpg'

# 4. Load images
content_image = load_image(content_path)
style_image = load_image(style_path)

# 5. Perform style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# 6. Save stylized output
output_path = 'stylized_output.png'
tf.keras.preprocessing.image.save_img(output_path, stylized_image[0])
print(f"Stylized image saved to {output_path}")

# 7. Wrap hub model in a Keras model to export
class StyleTransferModel(tf.keras.Model):
    def __init__(self, hub_model):
        super().__init__()
        self.hub = hub_model
    def call(self, inputs):
        content, style = inputs
        return self.hub(content, style)[0]

# 8. Build and save as .keras
input_content = tf.keras.layers.Input(shape=(256, 256, 3), name='content_input')
input_style = tf.keras.layers.Input(shape=(256, 256, 3), name='style_input')
output = StyleTransferModel(hub_model)([input_content, input_style])
transfer_keras = tf.keras.Model([input_content, input_style], output)
transfer_keras.save('style_transfer_model.keras')
print("Keras model saved to style_transfer_model.keras")

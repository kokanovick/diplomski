import os
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from pdf2image import convert_from_path
import pytesseract
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import cv2

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def fix_hyphenated_words(text):
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text

def extract_title_and_abstract(pdf_path):
    try:
        page = convert_from_path(pdf_path, dpi=600)
        text = pytesseract.image_to_string(page[0])
        text = fix_hyphenated_words(text)
        title_text = ""
        abstract_text = ""
        start_index = text.find("(54)")
        end_index = text.find('\n', start_index)
        while True:
            end_index = text.find('\n', end_index + 1)
            if end_index == -1:
                break
            if text[end_index:text.find('\n', end_index + 1)].strip():
                break
        title_text = text[start_index:end_index].replace("(54)", "").strip()
        abstract_index = text.find("ABSTRACT", end_index)
        start_index = text.find('\n\n', abstract_index) + 2
        end_index = text.find('\n\n', start_index)
        abstract_text += text[start_index:end_index].strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        title_text = ""
        abstract_text = ""
    combined_text = title_text + "\n" + abstract_text  
    return combined_text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]  
    return tokens

def preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def load_best_model():
    return load_model('C:/Users/SW6/Desktop/best_model.keras')

def extract_image_features(image_path, model):
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features.flatten()

def crop_schematic(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cY = int(M["m01"] / M["m00"])
        else:
            cY = 0
        if cY > height // 2:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_image_path = os.path.join(folder_path, "cropped_image.jpg")
    cv2.imwrite(cropped_image_path, cropped_image)
    return cropped_image_path

def generate_cam(image_path, model, last_conv_layer_name, classifier_layer_names):
    img_array = preprocess_image(image_path)
    model_input = model.input
    cam_model = Model(
        inputs=model_input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    conv_output, predictions = cam_model.predict(img_array)
      
    # Get the weights of the last classifier layer
    last_classifier_layer_name = classifier_layer_names[-1]
    class_weights = model.get_layer(last_classifier_layer_name).get_weights()[0]

    predicted_class = np.argmax(predictions[0])
    class_activation_weights = class_weights[:, predicted_class]

    # Compute CAM
    cam = np.zeros(dtype=np.float32, shape=conv_output.shape[1:3])   
    for i in range(min(class_activation_weights.shape[0], conv_output.shape[-1])):
        cam += class_activation_weights[i] * conv_output[0, :, :, i]
    
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # Overlay CAM on original image
    image = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    if heatmap.shape[2] != image.shape[2]:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_image = cv2.addWeighted(heatmap, 0.4, image, 0.6, 0)
    cam_image_path = os.path.join(os.path.dirname(image_path), "cam_image.jpg")
    cv2.imwrite(cam_image_path, superimposed_image)
    
    return cam_image_path


def process_pdfs_in_folder(folder_path):
    tokens = []
    image_features = []
    model = load_best_model()
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            combined_text = extract_title_and_abstract(pdf_path)
            tokens.append(preprocess_text(combined_text))
            page = convert_from_path(pdf_path, dpi=600)
            image_path = os.path.join(folder_path, "temp_image.jpg")
            page[0].save(image_path, 'JPEG')
            cropped_image_path = crop_schematic(image_path)
            cam_image_path = generate_cam(cropped_image_path, model, last_conv_layer_name='block5_conv4', classifier_layer_names=['dense'])
            features = extract_image_features(cam_image_path, model)
            image_features.append(features)
            os.remove(image_path)
            os.remove(cropped_image_path)
                
    return tokens, image_features

folder_path = 'C:/Users/SW6/Desktop/test'
tokens, image_features = process_pdfs_in_folder(folder_path)
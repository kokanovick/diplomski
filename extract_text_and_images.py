import os
import numpy as np
import torch
from torchvision import transforms
from pdf2image import convert_from_path
import pytesseract
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import cv2
from PIL import Image
import sys
import pathlib
from pytorch_grad_cam  import GradCAM
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator

sys.path.insert(0, 'C:/Users/SW6/Desktop/diplomski/yolov5')
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

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
    sentences = sent_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
        tokens = [token for token in tokens if token and token not in stop_words]
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
        processed_sentences.append(tokens)
    
    return processed_sentences

class WordEncoder(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim):
        super(WordEncoder, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        return output

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        weights = torch.tanh(self.attention(x))
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class SentenceEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(SentenceEncoder, self).__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        output, _ = self.gru(x)
        return output

class HAN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim):
        super(HAN, self).__init__()
        self.word_encoder = WordEncoder(embedding_matrix, hidden_dim)
        self.word_attention = Attention(hidden_dim)
        self.sentence_encoder = SentenceEncoder(hidden_dim)
        self.sentence_attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, documents):
        sentence_representations = []
        for sentences in documents:
            word_encoded_sentences = [self.word_encoder(sentence) for sentence in sentences]
            word_attended_sentences = [self.word_attention(sentence) for sentence in word_encoded_sentences]
            sentence_representations.append(torch.stack(word_attended_sentences))
        
        sentence_representations = torch.stack(sentence_representations)
        document_encoded = self.sentence_encoder(sentence_representations)
        document_attended = self.sentence_attention(document_encoded)
        
        output = self.fc(document_attended)
        return output

def load_embeddings(file_path, embedding_dim=150):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor

def load_yolov5_model(weights_path):
    model = torch.hub.load('C:/Users/SW6/Desktop/diplomski/yolov5', 'custom', source='local', path=weights_path)
    model.eval()
    return model

def load_vgg19_model():
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    return vgg19

def generate_cam(image_tensor, model, original_image):
    image_tensor.requires_grad = True
    target_layer = model.model.model[9].conv.conv
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
    heatmap = cv2.resize(grayscale_cam, (original_image.shape[1], original_image.shape[0]))     
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)     
    image_weight = 0.5
    cam_image = heatmap * image_weight + original_image * (1 - image_weight)
    cam_image = np.clip(cam_image, 0, 255).astype(np.uint8)
    Image.fromarray(cam_image).save('cam_result.jpg')  
    return cam_image, heatmap

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
    cropped_image_path = os.path.join(os.path.dirname(image_path), "cropped_image.jpg")
    cv2.imwrite(cropped_image_path, cropped_image)
    return cropped_image_path

def extract_image_features(image_tensor, vgg19_model):
    with torch.no_grad():
        features = vgg19_model(image_tensor).cpu().numpy()
    return features

def detect_components(image_path, model):
    results = model(image_path)
    probabilities = F.softmax(results, dim=1)
    return probabilities

def extract_text_features(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    tokens = preprocess_text(text)
    return tokens

def classify_image(detections, threshold=0.5):
    class_labels = {0: 'electronics', 1: 'flowchart'}
    class_scores = detections[0]
    predicted_class_index = class_scores.argmax().item()
    predicted_class_score = class_scores.max().item()
    if predicted_class_score >= threshold:
        return class_labels[predicted_class_index]
    else:
        return None   

def process_pdfs_in_folder(folder_path, yolov5_model, vgg19_model):
    tokens = []
    image_features = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            combined_text = extract_title_and_abstract(pdf_path)
            tokens.append(preprocess_text(combined_text))
    
            page = convert_from_path(pdf_path, dpi=600)
            image_path = os.path.join(folder_path, "temp_image.jpg")
            page[0].save(image_path, 'JPEG')
            cropped_image_path = crop_schematic(image_path)
            image_tensor = preprocess_image(cropped_image_path)
            
            detections = detect_components(image_tensor, yolov5_model)
            classification = classify_image(detections)
            
            if classification == 'flowchart':
                image_features.append(extract_text_features(cropped_image_path))
            elif classification == 'electronics':
                image_features.append(extract_image_features(image_tensor, vgg19_model))
            else:
                image_features.append(None)
            os.remove(cropped_image_path)
            os.remove(image_path)
                
    return tokens, image_features

weights_path = 'C:/Users/SW6/Desktop/diplomski/best.pt'
yolov5_model = load_yolov5_model(weights_path)
vgg19_model = load_vgg19_model()
folder_path = 'C:/Users/SW6/Desktop/test'
tokens, image_features = process_pdfs_in_folder(folder_path, yolov5_model, vgg19_model)

embedding_dim = 50
embedding_file_path = 'C:/Users/SW6/Desktop/test/glove.6B.50d.txt'
technet_embeddings = load_embeddings(embedding_file_path, embedding_dim)
all_tokens = [token for doc in tokens for sent in doc for token in sent]
vocab = build_vocab_from_iterator([all_tokens], specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, index in vocab.get_stoi().items():
    embedding_vector = technet_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
    else:
        embedding_matrix[index] = np.random.normal(size=(embedding_dim,))

indexed_documents = []
for doc in tokens:
    indexed_sentences = []
    for sent in doc:
        indexed_sent = [vocab[token] for token in sent]
        indexed_sentences.append(indexed_sent)
    indexed_documents.append(indexed_sentences)

def pad_documents(documents, padding_value=0):
    max_num_sentences = max(len(doc) for doc in documents)
    max_num_tokens = max(len(sent) for doc in documents for sent in doc)

    padded_documents = []
    for doc in documents:
        padded_sentences = []
        for sent in doc:
            padded_sent = sent + [padding_value] * (max_num_tokens - len(sent))
            padded_sentences.append(padded_sent)
        while len(padded_sentences) < max_num_sentences:
            padded_sentences.append([padding_value] * max_num_tokens)
        padded_documents.append(padded_sentences)

    return torch.tensor(padded_documents)

padded_documents = pad_documents(indexed_documents)
hidden_dim = len(indexed_sent)
model = HAN(embedding_matrix, hidden_dim)
output = model(padded_documents)
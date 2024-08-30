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
os.add_dll_directory(r'D:\OpenCV CUDA\install\x64\vc17\bin')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin')
import cv2
from PIL import Image
import sys
import pathlib
from pytorch_grad_cam  import GradCAM
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch_geometric.utils import from_networkx
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pickle

next(wordnet.words()) #to "materialize" LazyCorpusLoader
sys.path.insert(0, 'C:/Users/SW6/Desktop/diplomski/yolov5')
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text.lower())
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
    
    def forward(self, documents, lengths):
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

def load_embeddings(file_path, embedding_dim=50):
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
    image_tensor = preprocess(image).unsqueeze(0).to(device)
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
    
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
        gpu_blur_filter = cv2.cuda.createGaussianFilter(gpu_gray.type(), -1, (5, 5), 0)
        gpu_blurred = gpu_blur_filter.apply(gpu_gray)
        gpu_edges = cv2.cuda.createCannyEdgeDetector(50, 150)
        gpu_edges = gpu_edges.detect(gpu_blurred)
        edges = gpu_edges.download()
    else:
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
    cropped_image_path = os.path.join(os.path.dirname(image_path), "cropped_schematic.jpg")
    cv2.imwrite(cropped_image_path, cropped_image)
    return cropped_image_path

def crop_flowchart(image_path):
    image = cv2.imread(image_path)
    
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)    
        gray = gpu_gray.download()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_thresh = cv2.cuda_GpuMat()
        gpu_thresh.upload(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morph_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel)
        gpu_morph = morph_filter.apply(gpu_thresh)
        morph = gpu_morph.download()
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    largest_area = 0
    x_min, y_min, x_max, y_max = 0, 0, 0, 0
    for cnt in filtered_contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            x_min, y_min, w, h = cv2.boundingRect(cnt)
            x_max = x_min + w
            y_max = y_min + h

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_image_path = os.path.join(os.path.dirname(image_path), "cropped_flowchart.jpg")
    cv2.imwrite(cropped_image_path, cropped_image)
    return cropped_image_path


def extract_image_features(image_tensor, vgg19_model):
    vgg19_model.eval()
    with torch.no_grad():
        features = vgg19_model(image_tensor)
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

def process_single_pdf(pdf_path, index, yolov5_model, vgg19_model, device):
    tokens = []
    image_features = []
    flowchart_texts = []

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            combined_text = extract_title_and_abstract(pdf_path)
            tokens.append(preprocess_text(combined_text))
            image_path = os.path.join(temp_dir, "temp_image.jpg")
            page = convert_from_path(pdf_path, dpi=600)
            page[0].save(image_path, 'JPEG')
            cropped_image_path = crop_schematic(image_path)
            image_tensor = preprocess_image(cropped_image_path).to(device)
            detections = detect_components(image_tensor, yolov5_model)
            classification = classify_image(detections)

            if classification == 'flowchart':
                cropped_flowchart_path = crop_flowchart(cropped_image_path)
                flowchart_texts.append(extract_text_features(cropped_flowchart_path))
                image_features.append(0)
                os.remove(cropped_flowchart_path)
            elif classification == 'electronics':
                image_features.append(extract_image_features(image_tensor, vgg19_model))
            else:
                image_features.append(None)
            os.remove(cropped_image_path)
            
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return [], [], [], index

    return tokens, image_features, flowchart_texts, index

def process_pdfs_in_folder(base_folder_path, yolov5_model, vgg19_model):
    tokens = []
    image_features = []
    flowchart_texts = []

    with ThreadPoolExecutor() as executor:
        futures = []
        index = 0
        for class_folder in os.listdir(base_folder_path):
            class_path = os.path.join(base_folder_path, class_folder)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.pdf'):
                        pdf_path = os.path.join(class_path, file_name)
                        futures.append(executor.submit(process_single_pdf, pdf_path, index, yolov5_model, vgg19_model, device))
                        index += 1

        results = defaultdict(lambda: [[], [], []])
        for future in as_completed(futures):
            t, i_f, f_t, idx = future.result()
            results[idx][0].extend(t)
            results[idx][1].extend(i_f)
            results[idx][2].extend(f_t)

    ordered_results = sorted(results.items())
    for idx, result in ordered_results:
        tokens.extend(result[0])
        image_features.extend(result[1])
        flowchart_texts.extend(result[2])

    return tokens, image_features, flowchart_texts

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

weights_path = 'C:/Users/SW6/Desktop/diplomski/best.pt'
yolov5_model = load_yolov5_model(weights_path).to(device)
vgg19_model = load_vgg19_model().to(device)
base_folder_path = 'C:/Users/SW6/Desktop/diplomski/Baza'
tokens, image_features, flowchart_texts = process_pdfs_in_folder(base_folder_path, yolov5_model, vgg19_model)

embedding_dim = 50
embedding_file_path = 'C:/Users/SW6/Desktop/diplomski/glove.6B.50d.txt'
technet_embeddings = load_embeddings(embedding_file_path, embedding_dim)
all_tokens = [token for doc in tokens for sent in doc for token in sent]
all_tokens += [token for doc in flowchart_texts for sent in doc for token in sent]
vocab = build_vocab_from_iterator([all_tokens], specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, index in vocab.get_stoi().items():
    embedding_vector = technet_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
    else:
        embedding_matrix[index] = np.random.normal(size=(embedding_dim,))
        
def index_and_pad_documents(documents, vocab, padding_value=0):
    indexed_documents = []
    for doc in documents:
        indexed_sentences = []
        for sent in doc:
            indexed_sent = [vocab[token] for token in sent]
            indexed_sentences.append(indexed_sent)
        indexed_documents.append(indexed_sentences)
    
    max_num_sentences = 25#max(len(doc) for doc in indexed_documents)
    max_num_tokens = 15#max(len(sent) for doc in indexed_documents for sent in doc)
    
    padded_documents = []
    lengths = []
    for doc in indexed_documents:
        padded_sentences = []
        for sent in doc:
            if len(sent) > max_num_tokens:
                sent = sent[:max_num_tokens]  
            padded_sent = sent + [padding_value] * (max_num_tokens - len(sent)) 
            padded_sentences.append(padded_sent)
        lengths.append(len(padded_sentences))
        if len(padded_sentences) > max_num_sentences:
            padded_sentences = padded_sentences[:max_num_sentences]
        while len(padded_sentences) < max_num_sentences:
            padded_sentences.append([padding_value] * max_num_tokens)
        padded_documents.append(padded_sentences)
    
    return torch.tensor(padded_documents), lengths

def process_chunks(model, document_chunks, device):
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for chunk_documents, chunk_lengths in document_chunks:
            chunk_documents = chunk_documents.to(device)
            chunk_lengths = torch.tensor(chunk_lengths)
            
            with torch.cuda.amp.autocast():
                output = model(chunk_documents, chunk_lengths)
            all_outputs.append(output)
    
    return torch.cat(all_outputs, dim=0)

padded_documents, lengths = index_and_pad_documents(tokens, vocab)
def chunk_documents(padded_documents, lengths, chunk_size):
    for i in range(0, len(padded_documents), chunk_size):
        yield padded_documents[i:i + chunk_size], lengths[i:i + chunk_size]

chunk_size = 16  
document_chunks = list(chunk_documents(padded_documents, lengths, chunk_size))
hidden_dim = padded_documents.shape[2]
HAN_model = HAN(embedding_matrix, hidden_dim).to(device)
torch.cuda.empty_cache()
output = process_chunks(HAN_model, document_chunks, device)
han_features = output.detach().cpu()
processed_image_features = []
han_dim = han_features.shape[1]
for feature in image_features:
    if isinstance(feature, int) and feature == 0:
        processed_image_features.append(np.zeros((1, 1000)))
    else:
        processed_image_features.append(feature.detach().cpu().numpy())
        
processed_image_features = np.vstack(processed_image_features)
    
if processed_image_features.shape[1] != han_dim:
    if processed_image_features.shape[1] > han_dim:
        processed_image_features = processed_image_features[:, :han_dim]
    else:
        padding = np.zeros((processed_image_features.shape[0], han_dim - processed_image_features.shape[1]))
        processed_image_features = np.hstack((processed_image_features, padding))
unified_features = np.concatenate([han_features, processed_image_features], axis=1)
node_features = torch.tensor(unified_features, dtype=torch.float)
similarity_matrix = cosine_similarity(node_features)

similarity_scores = similarity_matrix.flatten()
plt.hist(similarity_scores, bins=50)
plt.xlabel('Kosinus sličnost')
plt.ylabel('Frekvencija')
plt.title('Distribucija rezultata kosinusne sličnosti')
plt.savefig("C:/Users/SW6/Desktop/diplomski/similiarity_scores_distribution.png")
plt.close()

G = nx.Graph()
num_nodes = len(node_features)
for i in range(num_nodes):
    G.add_node(i, features=node_features[i])

threshold = 0.2
edges = []
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if similarity_matrix[i, j] > threshold:
            edges.append((i, j))

scaler = StandardScaler()
normalized_features = scaler.fit_transform(node_features)
for i in range(num_nodes):
    G.nodes[i]['features'] = normalized_features[i]
data = from_networkx(G)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data.edge_index = edge_index
data.x = torch.tensor([G.nodes[i]['features'] for i in range(num_nodes)], dtype=torch.float)

class_folders = ['C:/Users/SW6/Desktop/diplomski/Baza/H01', 'C:/Users/SW6/Desktop/diplomski/Baza/H02', 'C:/Users/SW6/Desktop/diplomski/Baza/H03', 'C:/Users/SW6/Desktop/diplomski/Baza/H04', 'C:/Users/SW6/Desktop/diplomski/Baza/H05', 'C:/Users/SW6/Desktop/diplomski/Baza/H10']
class_labels = [0, 1, 2, 3, 4, 5] 

labels = []
for i, folder in enumerate(class_folders):
    num_docs = len([name for name in os.listdir(folder) if name.endswith('.pdf')])  
    labels.extend([class_labels[i]] * num_docs)

data.y = torch.tensor(labels, dtype=torch.long)

with open('C:/Users/SW6/Desktop/diplomski/data.pkl', 'wb') as f:
    pickle.dump(data, f)

class GraphSAGENetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GraphSAGENetwork, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)  
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)  
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)  
        x = self.fc(x)
        return x

def train(model, data, optimizer, criterion):
    model.train() 
    optimizer.zero_grad() 
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device) 
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.eval()  
    with torch.no_grad():  
        out = model(x, edge_index)
        _, predicted = torch.max(out[train_mask], dim=1)  
        correct = (predicted == y[train_mask]).sum().item() 
        accuracy = correct / train_mask.sum().item()  
    return loss.item(), accuracy

def validate(model, data, criterion):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        y = data.y.to(device)
        val_mask = data.val_mask.to(device)
        out = model(x, edge_index)
        val_loss = criterion(out[val_mask], y[val_mask]).item()
        _, pred = out[val_mask].max(dim=1)
        correct = pred.eq(y[val_mask]).sum().item()
        val_acc = correct / val_mask.sum().item()
    return val_loss, val_acc

def grid_search(data, learning_rates):
    best_lr = None
    best_val_loss = float('inf')
    best_val_acc = 0
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_ratio, val_ratio = 0.7, 0.15
    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)
    num_test = num_nodes - num_train - num_val
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        np.arange(num_nodes), labels, test_size=(num_val + num_test), stratify=labels, random_state=42
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels, test_size=(num_test / (num_val + num_test)), stratify=temp_labels, random_state=42
    )
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    model = GraphSAGENetwork(30, 128, len(set(data.y.tolist()))).to(device)

    for lr in learning_rates:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        num_epochs = 10
        print(f'Learning rate: {lr}')
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, data, optimizer, criterion)
            val_loss, val_acc = validate(model, data, criterion)
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr
            best_val_acc = val_acc

    print(f'Best LR: {best_lr}, Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}')
    return best_lr, best_val_loss, best_val_acc

learning_rates = [0.001, 0.01, 0.1, 0.0001, 0.00001]
criterion = nn.CrossEntropyLoss()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=5)

def test(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        y = data.y.to(device)
        test_mask = data.test_mask.to(device)
        out = model(x, edge_index)
        pred = out[test_mask].argmax(dim=1)
        y_true = y[test_mask].cpu()
        pred = pred.cpu()
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, average='weighted')
        precision = precision_score(y_true, pred, average='weighted')
        recall = recall_score(y_true, pred, average='weighted')

    return acc, f1, precision, recall, y_true, pred

def plot_confusion_matrix(y_true, y_pred, classes, part):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predviđena oznaka')
    plt.ylabel('Prava oznaka')
    plt.title('Matrica zabune')
    plt.savefig("/content/confusion_matrix" + str(part) + ".png")
    plt.close()

classes = ["H01", "H02", "H03", "H04", "H05", "H10"]
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
all_test_metrics = []

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_accuracies = []
test_f1_scores = []
test_precisions = []
test_recalls = []
best_lr, best_val_loss, best_val_acc = grid_search(data, learning_rates)


fold = 1
for train_index, test_index in skf.split(np.arange(num_nodes), labels):
    print(f'Fold {fold}/{n_splits}')
    train_index, val_index, train_labels, val_labels = train_test_split(
        train_index, [labels[i] for i in train_index], test_size=(1/n_splits), stratify=[labels[i] for i in train_index], random_state=42
    )
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_index] = True
    data.val_mask[val_index] = True
    data.test_mask[test_index] = True
    data.y = torch.tensor(labels, dtype=torch.long)
    model = GraphSAGENetwork(30, 128, len(set(data.y.tolist()))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    early_stopping = EarlyStopping(patience=5)
    fold_train_losses = []
    fold_train_accuracies = []
    fold_val_losses = []
    fold_val_accuracies = []
    for epoch in range(90):
        train_loss, train_acc = train(model, data, optimizer, criterion)
        val_loss, val_acc = validate(model, data, criterion)
        early_stopping(val_loss)

        fold_train_losses.append(train_loss)
        fold_train_accuracies.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accuracies.append(val_acc)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    train_losses.append(fold_train_losses)
    train_accuracies.append(fold_train_accuracies)
    val_losses.append(fold_val_losses)
    val_accuracies.append(fold_val_accuracies)
    test_acc, test_f1, test_prec, test_recall, y_true, y_pred = test(model, data)
    test_accuracies.append(test_acc)
    test_f1_scores.append(test_f1)
    test_precisions.append(test_prec)
    test_recalls.append(test_recall)
    plot_confusion_matrix(y_true, y_pred, classes, fold)
    fold += 1

max_epochs = max(len(fold_losses) for fold_losses in train_losses)

def pad_list(lst, max_len, pad_value=None):
    return lst + [pad_value] * (max_len - len(lst))

avg_train_losses = np.mean([pad_list(fold_losses, max_epochs, np.nan) for fold_losses in train_losses], axis=0)
avg_train_accuracies = np.mean([pad_list(fold_accs, max_epochs, np.nan) for fold_accs in train_accuracies], axis=0)
avg_val_losses = np.mean([pad_list(fold_losses, max_epochs, np.nan) for fold_losses in val_losses], axis=0)
avg_val_accuracies = np.mean([pad_list(fold_accs, max_epochs, np.nan) for fold_accs in val_accuracies], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses, label='Prosječan trening gubitak')
plt.plot(avg_val_losses, label='Prosječan validacijski gubitak')
plt.xlabel('Epoha')
plt.ylabel('Gubitak')
plt.title('Prosječni trening i validacijski gubitci kroz nabore')
plt.legend()
plt.savefig("/content/average_train_valid_loss.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(avg_train_accuracies, label='Prosječna trening točnost')
plt.plot(avg_val_accuracies, label='Prosječna validacijska točnost')
plt.xlabel('Epoha')
plt.ylabel('Točnost')
plt.title('Prosječna trening i validacijska točnost kroz nabore')
plt.legend()
plt.savefig("/content/average_valid_acc.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(test_accuracies, label='Test točnost')
plt.plot(test_f1_scores, label='Test F1 rezultat')
plt.plot(test_precisions, label='Test preciznost')
plt.plot(test_recalls, label='Test odziv')
plt.xlabel('Nabor')
plt.ylabel('Rezultat')
plt.title('Promatrane metrike kroz nabore')
plt.legend()
plt.savefig("/content/test_metrics.png")
plt.show()
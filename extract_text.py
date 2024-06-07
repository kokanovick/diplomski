import pytesseract
from pdf2image import convert_from_path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re

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
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Denoising: Remove non-alphanumeric characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    
    # Remove empty strings resulted from denoising
    tokens = [token for token in tokens if token]
    
    # Stop-word Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]  
    return tokens

pdf_path = 'C:/Users/SW6/Desktop/diplomski/Baza/US20230230744A1.pdf_page1.pdf'
combined_text = extract_title_and_abstract(pdf_path)
tokens = preprocess_text(combined_text)
import spacy
from multiprocessing import Manager

class TokenLogger:
    def __init__(self):
        manager = Manager()
        self.token_count = manager.Value('i', 0)  # Shared integer using Manager
        self.lock = manager.Lock()  # Lock for synchronization using Manager
        self.nlp = spacy.load('en_core_web_sm')

    def log_tokens(self, text):
        doc = self.nlp(text)
        with self.lock:
            self.token_count.value += len(doc)
        return text

    def get_token_count(self):
        with self.lock:
            return self.token_count.value

from preprocessing import nlp
class TokenLogger:
    def __init__(self):
        self.token_count = 0

    def log_tokens(self, text):
        self.token_count += len(nlp(text))
        return text

    def get_token_count(self):
        return self.token_count
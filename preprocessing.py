from token_logger import *
from data_extraction import read_resume
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os



nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def extract_features(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
    experience = [ent.text for ent in doc.ents if ent.label_ == 'EXPERIENCE']
    location = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    salary = [ent.text for ent in doc.ents if ent.label_ == 'MONEY']
    return skills, experience, location, salary


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_similarity(job_description, resume_texts):
    job_embedding = model.encode([job_description])
    resume_embeddings = model.encode(resume_texts)
    similarities = cosine_similarity(job_embedding, resume_embeddings)
    return similarities[0]

def rank_candidates(resumes, job_description):
    resume_texts = [preprocess_text(read_resume(resume)) for resume in resumes]
    similarities = calculate_similarity(job_description, resume_texts)
    ranked_indices = similarities.argsort()[-3:][::-1]
    return [(resumes[i], similarities[i]) for i in ranked_indices]


def get_all_resumes(directory_path):
    supported_formats = ('.pdf', '.docx', '.txt')
    resumes = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(supported_formats)]
    return resumes

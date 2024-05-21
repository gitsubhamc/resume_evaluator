import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from data_extraction import read_resume

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

def calculate_similarity(job_description, resume_texts, job_skills, job_experience, job_location, job_salary):
    job_embedding = model.encode([job_description])
    resume_embeddings = model.encode(resume_texts)
    
    text_similarities = cosine_similarity(job_embedding, resume_embeddings)
    
    similarities = []
    for i in range(len(resume_texts)):
        resume_skills, resume_experience, resume_location, resume_salary = extract_features(resume_texts[i])
        
        skills_similarity = 0.0
        if job_skills and resume_skills:
            skills_similarity = cosine_similarity([job_skills], [resume_skills])[0][0]
        
        experience_similarity = 0.0
        if job_experience and resume_experience:
            experience_similarity = cosine_similarity([job_experience], [resume_experience])[0][0]
        
        location_similarity = 0.0
        if job_location and resume_location:
            location_similarity = cosine_similarity([job_location], [resume_location])[0][0]
        
        salary_similarity = 0.0
        if job_salary and resume_salary:
            salary_similarity = cosine_similarity([job_salary], [resume_salary])[0][0]
        
        total_similarity = 0.4 * text_similarities[0][i] + 0.15 * skills_similarity + 0.15 * experience_similarity + 0.15 * location_similarity + 0.15 * salary_similarity
        
        similarities.append(total_similarity)
    
    return similarities


def rank_candidates(resumes, job_description):
    resume_texts = [preprocess_text(read_resume(resume)) for resume in resumes]
    
    job_skills, job_experience, job_location, job_salary = extract_features(job_description)
    
    similarities = calculate_similarity(job_description, resume_texts, job_skills, job_experience, job_location, job_salary)
    ranked_candidates = list(zip(resumes, similarities))
    ranked_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_candidates[:3]

def get_all_resumes(directory_path):
    supported_formats = ('.pdf', '.docx', '.txt')
    resumes = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(supported_formats)]
    return resumes

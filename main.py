from token_logger import *
from preprocessing import get_all_resumes,rank_candidates
from data_extraction import read_resume
token_logger = TokenLogger()

def process_resume_with_logging(file_path):
    text = read_resume(file_path)
    token_logger.log_tokens(text)
    return text


directory_path = './../resumes'
resumes = get_all_resumes(directory_path)

# Process resumes and rank candidates
processed_resumes = [process_resume_with_logging(resume) for resume in resumes]
job_description = "best fit for nlp roles"
ranked_candidates = rank_candidates(resumes, job_description)

# Print top 3 candidates and their similarity scores
for resume, score in ranked_candidates:
    print(f"Resume: {resume}, Similarity Score: {score}")

# Print total tokens used
print("Total tokens used:", token_logger.get_token_count())

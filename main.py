from token_logger import TokenLogger
from preprocessing import get_all_resumes, rank_candidates
from data_extraction import read_resume
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager



def process_resume_with_logging(file_path, token_logger):
    text = read_resume(file_path)
    token_logger.log_tokens(text)
    return text

def process_batch(batch, job_description, token_logger):
    processed_resumes = [process_resume_with_logging(resume, token_logger) for resume in batch]
    return rank_candidates(batch, job_description)

if __name__ == '__main__':
    manager = Manager()
    token_logger = TokenLogger()

    directory_path = './../resumes'
    resumes = get_all_resumes(directory_path)
    job_description = "best fit for nlp roles"
    batch_size = 100

    # Process resumes in batches
    ranked_candidates = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, len(resumes), batch_size):
            batch = resumes[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch, job_description, token_logger))
        
        for future in futures:
            ranked_candidates.extend(future.result())

    # Sort and print top 3 candidates and their similarity scores
    ranked_candidates = sorted(ranked_candidates, key=lambda x: x[1], reverse=True)[:3]
    for resume, score in ranked_candidates:
        print(f"Resume: {resume}, Similarity Score: {score}")

    # Print total tokens used
    print("Total tokens used:", token_logger.get_token_count())

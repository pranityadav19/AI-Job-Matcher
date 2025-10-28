"""
Job Matcher Algorithm using NLP Techniques
Combines TF-IDF, Word Embeddings, and LDA for optimal job matching 
"""

import json
import numpy as np
import re
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn for TF-IDF and LDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

# Sentence transformers for semantic embeddings
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class JobMatcher:
    """
    A comprehensive job matching system using multiple NLP techniques:
    - TF-IDF for keyword-based similarity
    - Sentence embeddings for semantic similarity
    - LDA for topic modeling
    """
    
    def __init__(self, jobs_file_path: str, n_topics: int = 20):
        """
        Initialize the job matcher with job data
        
        Args:
            jobs_file_path: Path to JSON file containing job data
            n_topics: Number of topics for LDA model
        """
        self.jobs = self._load_jobs(jobs_file_path)
        self.n_topics = n_topics
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words for job descriptions
        self.stop_words.update([
            'job', 'work', 'position', 'role', 'candidate', 'experience',
            'team', 'company', 'opportunity', 'responsibilities', 'requirements'
        ])
        
        # Initialize models (will be trained on first use)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.lda_model = None
        self.lda_vectorizer = None
        self.job_topics = None
        self.embedding_model = None
        self.job_embeddings = None
        
        # NEW: Cache for resume embeddings
        self._cached_resume_text = None
        self._cached_resume_embedding = None
        
        # Preprocess job data
        self.processed_jobs = self._preprocess_jobs()
        
        print(f"[OK] Loaded {len(self.jobs)} jobs successfully")
        
    def _load_jobs(self, file_path: str) -> List[Dict]:
        """
        Load job data from JSON file with validation
        
        IMPROVED: Now validates data and handles missing fields
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Jobs file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
        
        # Validate and clean jobs
        required_fields = ['title', 'description']
        validated_jobs = []
        skipped = 0
        
        for i, job in enumerate(jobs):
            # Ensure required fields exist and are not empty
            if not all(field in job and job[field] for field in required_fields):
                print(f"Warning: Skipping job {i+1} - missing required fields (title or description)")
                skipped += 1
                continue
            
            # Add defaults for optional fields
            job.setdefault('company', 'Unknown Company')
            job.setdefault('location', 'Unknown Location')
            job.setdefault('salary', 'Not specified')
            job.setdefault('url', '')
            job.setdefault('id', str(i))
            
            validated_jobs.append(job)
        
        if skipped > 0:
            print(f"Warning: Skipped {skipped} invalid job(s)")
        
        if len(validated_jobs) == 0:
            raise ValueError("No valid jobs found in the file")
        
        return validated_jobs
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        IMPROVED: Now preserves technical terms and numbers
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # IMPROVED: Preserve common technical terms before removing special chars
        # These are important for job matching!
        text = re.sub(r'c\+\+', 'cplusplus', text)
        text = re.sub(r'c#', 'csharp', text)
        text = re.sub(r'\.net', 'dotnet', text)
        text = re.sub(r'node\.js', 'nodejs', text)
        text = re.sub(r'react\.js', 'reactjs', text)
        text = re.sub(r'vue\.js', 'vuejs', text)
        
        # IMPROVED: Keep alphanumeric combinations (Python3, AWS-S3, etc.)
        # This regex keeps numbers attached to letters
        text = re.sub(r'([a-zA-Z]+)[\-\.]?(\d+)', r'\1\2', text)
        
        # IMPROVED: Now keeps numbers for experience years and versions
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        lemmatized = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return lemmatized
    
    def _preprocess_jobs(self) -> List[Dict]:
        """Preprocess all jobs for analysis"""
        processed = []
        
        for job in self.jobs:
            # Combine relevant fields for matching
            full_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('company', '')} {job.get('location', '')}"
            
            # Clean text
            cleaned = self._clean_text(full_text)
            
            # Tokenize and lemmatize
            tokens = self._tokenize_and_lemmatize(cleaned)
            
            processed.append({
                'original': job,
                'cleaned_text': cleaned,
                'tokens': tokens,
                'token_string': ' '.join(tokens)
            })
        
        return processed
    
    def _train_tfidf(self):
        """Train TF-IDF vectorizer on job corpus"""
        print("Training TF-IDF model...")
        
        corpus = [job['token_string'] for job in self.processed_jobs]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        print(f"[OK] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _train_lda(self):
        """Train LDA model for topic modeling"""
        print("Training LDA model...")
        
        corpus = [job['token_string'] for job in self.processed_jobs]
        
        # Use CountVectorizer for LDA (it requires count data, not TF-IDF)
        self.lda_vectorizer = CountVectorizer(
            max_features=3000,
            min_df=5,
            max_df=0.7
        )
        
        doc_term_matrix = self.lda_vectorizer.fit_transform(corpus)
        
        # Train LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            n_jobs=-1,
            max_iter=20
        )
        
        self.job_topics = self.lda_model.fit_transform(doc_term_matrix)
        print(f"[OK] LDA model trained with {self.n_topics} topics")
        
    def _train_embeddings(self):
        """Generate embeddings for all jobs using sentence transformers"""
        print("Generating job embeddings (this may take a moment)...")
        
        # Use a pre-trained sentence transformer model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for job descriptions
        job_texts = [job['cleaned_text'] for job in self.processed_jobs]
        self.job_embeddings = self.embedding_model.encode(
            job_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"[OK] Generated embeddings with shape: {self.job_embeddings.shape}")
    
    def train_models(self):
        """Train all NLP models"""
        print("\n" + "="*60)
        print("TRAINING NLP MODELS")
        print("="*60 + "\n")
        
        self._train_tfidf()
        self._train_lda()
        self._train_embeddings()
        
        print("\n" + "="*60)
        print("[OK] ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*60 + "\n")
    
    def _preprocess_resume(self, resume_text: str) -> Dict:
        """Preprocess resume text"""
        cleaned = self._clean_text(resume_text)
        tokens = self._tokenize_and_lemmatize(cleaned)
        
        return {
            'cleaned_text': cleaned,
            'tokens': tokens,
            'token_string': ' '.join(tokens)
        }
    
    def _calculate_tfidf_similarity(self, resume_processed: Dict) -> np.ndarray:
        """Calculate TF-IDF cosine similarity scores"""
        resume_vector = self.tfidf_vectorizer.transform([resume_processed['token_string']])
        similarities = cosine_similarity(resume_vector, self.tfidf_matrix).flatten()
        return similarities
    
    def _calculate_lda_similarity(self, resume_processed: Dict) -> np.ndarray:
        """Calculate LDA topic similarity scores"""
        # Transform resume to topic distribution
        resume_vector = self.lda_vectorizer.transform([resume_processed['token_string']])
        resume_topics = self.lda_model.transform(resume_vector)
        
        # Calculate cosine similarity between topic distributions
        similarities = cosine_similarity(resume_topics, self.job_topics).flatten()
        return similarities
    
    def _calculate_embedding_similarity(self, resume_processed: Dict) -> np.ndarray:
        """
        Calculate semantic similarity using embeddings
        
        IMPROVED: Now caches resume embedding to avoid recomputation
        """
        # Check if we need to recompute the resume embedding
        if (self._cached_resume_text != resume_processed['cleaned_text'] or 
            self._cached_resume_embedding is None):
            # Generate new embedding
            self._cached_resume_embedding = self.embedding_model.encode(
                [resume_processed['cleaned_text']],
                convert_to_numpy=True
            )
            self._cached_resume_text = resume_processed['cleaned_text']
        
        # Use cached embedding
        resume_embedding = self._cached_resume_embedding
        
        # Calculate cosine similarity
        similarities = cosine_similarity(resume_embedding, self.job_embeddings).flatten()
        return similarities
    
    def match_jobs(
        self,
        resume_text: str,
        top_k: int = 20,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Match jobs to a resume using hybrid approach
        
        Args:
            resume_text: Text content of the resume
            top_k: Number of top matches to return
            weights: Dictionary of weights for each method
                    {'tfidf': 0.3, 'lda': 0.2, 'embedding': 0.5}
        
        Returns:
            List of top matching jobs with scores
        """
        # Ensure models are trained
        if self.tfidf_matrix is None:
            print("Models not trained. Training now...")
            self.train_models()
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'tfidf': 0.25,      # Keyword matching
                'lda': 0.25,        # Topic similarity
                'embedding': 0.50   # Semantic similarity (most important)
            }
        
        print("Preprocessing resume...")
        resume_processed = self._preprocess_resume(resume_text)
        
        print("Calculating similarities...")
        # Calculate similarities using each method
        tfidf_scores = self._calculate_tfidf_similarity(resume_processed)
        lda_scores = self._calculate_lda_similarity(resume_processed)
        embedding_scores = self._calculate_embedding_similarity(resume_processed)
        
        # Normalize scores to 0-1 range
        def normalize(scores):
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score > 0:
                return (scores - min_score) / (max_score - min_score)
            return scores
        
        tfidf_scores_norm = normalize(tfidf_scores)
        lda_scores_norm = normalize(lda_scores)
        embedding_scores_norm = normalize(embedding_scores)
        
        # Calculate weighted combined score
        combined_scores = (
            weights['tfidf'] * tfidf_scores_norm +
            weights['lda'] * lda_scores_norm +
            weights['embedding'] * embedding_scores_norm
        )
        
        # Get top k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            job = self.processed_jobs[idx]['original']
            results.append({
                'job': job,
                'overall_score': float(combined_scores[idx]),
                'tfidf_score': float(tfidf_scores_norm[idx]),
                'lda_score': float(lda_scores_norm[idx]),
                'embedding_score': float(embedding_scores_norm[idx])
            })
        
        print(f"[OK] Found top {top_k} matching jobs\n")
        return results
    
    def get_top_lda_topics(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get top words for each LDA topic
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            List of topics, each containing top words and their weights
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet")
        
        feature_names = self.lda_vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_indices]
            topics.append(top_words)
        
        return topics
    
    def print_topic_summary(self, n_words: int = 10):
        """Print a summary of discovered topics"""
        topics = self.get_top_lda_topics(n_words)
        
        print(f"\n{'='*60}")
        print(f"LDA TOPICS DISCOVERED ({self.n_topics} topics)")
        print(f"{'='*60}\n")
        
        for idx, topic_words in enumerate(topics):
            words = [word for word, _ in topic_words]
            print(f"Topic {idx + 1}: {', '.join(words)}")
        
        print(f"\n{'='*60}\n")
    
    def explain_match(self, resume_text: str, job_idx: int, top_n: int = 10) -> Dict:
        """
        Explain why a specific job matched with the resume
        
        Args:
            resume_text: The resume text
            job_idx: Index of the job in processed_jobs
            top_n: Number of top matching keywords to return
            
        Returns:
            Dictionary with explanation details
        """
        # Preprocess resume
        resume_processed = self._preprocess_resume(resume_text)
        job_processed = self.processed_jobs[job_idx]
        
        # Find common tokens
        resume_tokens = set(resume_processed['tokens'])
        job_tokens = set(job_processed['tokens'])
        common_tokens = resume_tokens & job_tokens
        
        # Get TF-IDF feature scores
        resume_vec = self.tfidf_vectorizer.transform([resume_processed['token_string']])
        job_vec = self.tfidf_matrix[job_idx]
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        resume_features = resume_vec.toarray()[0]
        job_features = job_vec.toarray()[0]
        
        # Find matching features with their importance scores
        matching_features = []
        for i, (r_score, j_score) in enumerate(zip(resume_features, job_features)):
            if r_score > 0 and j_score > 0:
                # Both resume and job have this feature
                importance = min(r_score, j_score) * (r_score + j_score)
                matching_features.append((feature_names[i], importance))
        
        # Sort by importance
        matching_features.sort(key=lambda x: x[1], reverse=True)
        
        # Get top keywords
        top_keywords = [kw for kw, _ in matching_features[:top_n]]
        
        # Calculate match percentage
        if len(resume_tokens) > 0:
            token_overlap = len(common_tokens) / len(resume_tokens)
        else:
            token_overlap = 0
        
        # Determine match strength
        if token_overlap > 0.3:
            strength = "Strong"
        elif token_overlap > 0.15:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        return {
            'top_keywords': top_keywords,
            'common_tokens': sorted(list(common_tokens))[:20],
            'token_overlap_percent': token_overlap * 100,
            'match_strength': strength,
            'total_common_tokens': len(common_tokens)
        }


def save_results_to_json(results: List[Dict], output_file: str):
    """Save matching results to a JSON file"""
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # If output_file doesn't include a directory, put it in results/
    if not os.path.dirname(output_file):
        output_file = os.path.join(results_dir, output_file)
    
    # Prepare data for JSON serialization
    output_data = []
    for result in results:
        output_data.append({
            'job_id': result['job']['id'],
            'title': result['job']['title'],
            'company': result['job']['company'],
            'location': result['job']['location'],
            'salary': result['job']['salary'],
            'url': result['job']['url'],
            'scores': {
                'overall': result['overall_score'],
                'tfidf': result['tfidf_score'],
                'lda': result['lda_score'],
                'embedding': result['embedding_score']
            },
            'description': result['job']['description'][:500] + "..."  # Truncate for readability
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Results saved to {output_file}")


def save_results_to_text(results: List[Dict], output_file: str, resume_file: str = None, resume_text: str = None, matcher = None):
    """
    Save matching results to a readable text file with summary and explanations
    
    Args:
        results: List of job matching results
        output_file: Path to output text file
        resume_file: Name of the resume file used (optional)
        resume_text: The actual resume text for generating explanations (optional)
        matcher: JobMatcher instance for generating explanations (optional)
    """
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # If output_file doesn't include a directory, put it in results/
    if not os.path.dirname(output_file):
        output_file = os.path.join(results_dir, output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("JOB MATCHING RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Metadata
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if resume_file:
            f.write(f"Resume File: {resume_file}\n")
        f.write(f"Total Matches: {len(results)}\n")
        f.write("\n")
        
        # TOP 3 SUMMARY
        f.write("="*80 + "\n")
        f.write("TOP 3 JOB MATCHES - QUICK SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(results[:3], 1):
            job = result['job']
            f.write(f"{i}. {job['title']}\n")
            f.write(f"   Company: {job['company']}\n")
            f.write(f"   Location: {job['location']}\n")
            f.write(f"   Match Score: {result['overall_score']:.2%}\n")
            f.write(f"   Salary: {job.get('salary', 'Not specified')}\n")
            
            # Show job requirements in summary
            if 'job_requirements' in result and result['job_requirements']:
                job_reqs = result['job_requirements']
                reqs_text = []
                if 'min_years' in job_reqs:
                    reqs_text.append(f"{job_reqs['min_years']}+ years exp")
                if 'required_degree' in job_reqs:
                    reqs_text.append(f"{job_reqs['required_degree'].title()} degree")
                if reqs_text:
                    f.write(f"   Requirements: {', '.join(reqs_text)}\n")
                    
                    # Show if candidate qualifies
                    if 'candidate_qualifies' in result and result['candidate_qualifies'] is not None:
                        if result['candidate_qualifies']:
                            f.write(f"   Status: [YOU QUALIFY]\n")
                        else:
                            f.write(f"   Status: [DOES NOT MEET REQUIREMENTS]\n")
            
            f.write(f"   URL: {job.get('url', 'N/A')}\n")
            
            # Add explanation if available
            if matcher and resume_text:
                try:
                    # Find the job index in matcher's processed_jobs
                    job_idx = next((idx for idx, pj in enumerate(matcher.processed_jobs) 
                                  if pj['original'].get('id') == job.get('id')), None)
                    if job_idx is not None:
                        explanation = matcher.explain_match(resume_text, job_idx, top_n=8)
                        f.write(f"   WHY IT MATCHED:\n")
                        f.write(f"     - Match Strength: {explanation['match_strength']}\n")
                        f.write(f"     - Key Matches: {', '.join(explanation['top_keywords'][:5])}\n")
                        f.write(f"     - Token Overlap: {explanation['token_overlap_percent']:.1f}%\n")
                except:
                    pass
            
            f.write("\n")
        
        # DETAILED RESULTS
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS - ALL MATCHES\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            job = result['job']
            
            f.write(f"MATCH #{i}\n")
            f.write("-"*80 + "\n")
            f.write(f"Job Title: {job['title']}\n")
            f.write(f"Company: {job['company']}\n")
            f.write(f"Location: {job['location']}\n")
            f.write(f"Salary: {job.get('salary', 'Not specified')}\n")
            f.write("\n")
            
            # Show job requirements if available
            if 'job_requirements' in result:
                job_reqs = result['job_requirements']
                if job_reqs:
                    f.write(f"JOB REQUIREMENTS:\n")
                    if 'min_years' in job_reqs:
                        f.write(f"  Minimum Experience: {job_reqs['min_years']} years\n")
                    if 'required_degree' in job_reqs:
                        f.write(f"  Required Degree: {job_reqs['required_degree'].title()}\n")
                else:
                    f.write(f"JOB REQUIREMENTS:\n")
                    f.write(f"  No specific requirements detected in job description\n")
            else:
                f.write(f"JOB REQUIREMENTS:\n")
                f.write(f"  No requirements parsed\n")
            
            # ALWAYS show qualification status
            if 'candidate_qualifies' in result and result['candidate_qualifies'] is not None:
                if result['candidate_qualifies']:
                    f.write(f"  YOUR STATUS: [YOU QUALIFY]\n")
                else:
                    f.write(f"  YOUR STATUS: [DOES NOT MEET REQUIREMENTS]\n")
            else:
                f.write(f"  YOUR STATUS: [REQUIREMENTS NOT SPECIFIED - LIKELY QUALIFY]\n")
            
            f.write("\n")
            
            f.write(f"MATCH SCORES:\n")
            f.write(f"  Overall Score:    {result['overall_score']:.4f} ({result['overall_score']:.2%})\n")
            f.write(f"  TF-IDF Score:     {result['tfidf_score']:.4f} (keyword matching)\n")
            f.write(f"  LDA Score:        {result['lda_score']:.4f} (topic similarity)\n")
            f.write(f"  Embedding Score:  {result['embedding_score']:.4f} (semantic similarity)\n")
            f.write("\n")
            
            # Add detailed explanation
            if matcher and resume_text:
                try:
                    job_idx = next((idx for idx, pj in enumerate(matcher.processed_jobs) 
                                  if pj['original'].get('id') == job.get('id')), None)
                    if job_idx is not None:
                        explanation = matcher.explain_match(resume_text, job_idx, top_n=10)
                        
                        f.write(f"WHY THIS JOB MATCHED:\n")
                        f.write(f"  Match Strength: {explanation['match_strength']}\n")
                        f.write(f"  Token Overlap: {explanation['token_overlap_percent']:.1f}% ")
                        f.write(f"({explanation['total_common_tokens']} common terms)\n\n")
                        
                        f.write(f"  Top Matching Keywords/Phrases:\n")
                        for idx, keyword in enumerate(explanation['top_keywords'][:10], 1):
                            f.write(f"    {idx}. {keyword}\n")
                        
                        if len(explanation['common_tokens']) > 0:
                            f.write(f"\n  All Matching Terms:\n")
                            f.write(f"    {', '.join(explanation['common_tokens'][:30])}")
                            if len(explanation['common_tokens']) > 30:
                                f.write(f"... and {len(explanation['common_tokens']) - 30} more")
                            f.write("\n")
                        
                        f.write(f"\n  Score Breakdown:\n")
                        if result['tfidf_score'] > 0.7:
                            f.write(f"    - High keyword match: Your resume shares many specific terms\n")
                        elif result['tfidf_score'] > 0.4:
                            f.write(f"    - Moderate keyword match: Some shared terminology\n")
                        else:
                            f.write(f"    - Lower keyword match: Different specific terms used\n")
                        
                        if result['embedding_score'] > 0.7:
                            f.write(f"    - High semantic match: Similar meaning and context\n")
                        elif result['embedding_score'] > 0.4:
                            f.write(f"    - Moderate semantic match: Related concepts and ideas\n")
                        else:
                            f.write(f"    - Lower semantic match: Different domain or focus\n")
                        
                        f.write("\n")
                except Exception as e:
                    f.write(f"  (Could not generate detailed explanation)\n\n")
            
            f.write(f"Job Description:\n")
            description = job.get('description', 'No description available')
            # Wrap text at 80 characters
            words = description.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 <= 78:
                    line += word + " "
                else:
                    f.write(f"  {line}\n")
                    line = word + " "
            if line:
                f.write(f"  {line}\n")
            f.write("\n")
            
            f.write(f"Application URL: {job.get('url', 'N/A')}\n")
            f.write("\n" + "="*80 + "\n\n")
        
        # Footer
        f.write("\nEND OF RESULTS\n")
        f.write("="*80 + "\n")
    
    print(f"[OK] Text results saved to {output_file}")


def print_results(results: List[Dict], top_n: int = 10):
    """Print matching results in a formatted way"""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MATCHING JOBS")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:top_n], 1):
        job = result['job']
        print(f"{i}. {job['title']} at {job['company']}")
        print(f"   Location: {job['location']}")
        print(f"   Salary: {job.get('salary', 'Not specified')}")
        
        # Show job requirements and qualification status
        if 'job_requirements' in result and result['job_requirements']:
            job_reqs = result['job_requirements']
            reqs_text = []
            if 'min_years' in job_reqs:
                reqs_text.append(f"{job_reqs['min_years']}+ yrs")
            if 'required_degree' in job_reqs:
                reqs_text.append(f"{job_reqs['required_degree'].title()}")
            if reqs_text:
                print(f"   Requirements: {', '.join(reqs_text)}")
        
        # Show qualification status
        if 'candidate_qualifies' in result and result['candidate_qualifies'] is not None:
            if result['candidate_qualifies']:
                print(f"   Status: [YOU QUALIFY]")
            else:
                print(f"   Status: [DOES NOT MEET REQUIREMENTS]")
        
        print(f"   Overall Score: {result['overall_score']:.4f}")
        print(f"   - TF-IDF: {result['tfidf_score']:.4f}")
        print(f"   - LDA: {result['lda_score']:.4f}")
        print(f"   - Embedding: {result['embedding_score']:.4f}")
        print(f"   URL: {job['url']}")
        print(f"   {'-'*76}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    print("Job Matcher Example")
    print("="*60)
    
    # Initialize matcher
    matcher = JobMatcher('job_results.json', n_topics=20)
    
    # Train models
    matcher.train_models()
    
    # Show discovered topics
    matcher.print_topic_summary(n_words=8)
    
    # Example resume (you would load this from a file)
    example_resume = """
    Senior Data Scientist with 7 years of experience in machine learning, 
    deep learning, and statistical analysis. Expert in Python, R, TensorFlow, 
    PyTorch, and scikit-learn. Experience with large-scale data processing 
    using Spark and cloud platforms (AWS, GCP). Strong background in NLP, 
    computer vision, and predictive modeling. PhD in Computer Science with 
    focus on artificial intelligence. Published researcher with multiple 
    papers in top-tier conferences.
    """
    
    # Match jobs
    results = matcher.match_jobs(example_resume, top_k=20)
    
    # Print results
    print_results(results, top_n=10)
    
    # Save results
    save_results_to_json(results, 'matched_jobs.json')
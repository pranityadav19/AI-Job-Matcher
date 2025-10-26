#!/usr/bin/env python3
"""
Main Script for Job Matcher System
Easy-to-use interface for matching resumes to jobs
"""

import argparse
import sys
import os
from JobMatcher import JobMatcher, print_results, save_results_to_json, save_results_to_text
from ResumeExtractor import ResumeExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Match jobs to your resume using NLP techniques',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match jobs to a resume
  python JobMain.py --resume my_resume.pdf --jobs job_results.json --top 20

  # Customize scoring weights
  python JobMain.py --resume resume.docx --jobs jobs.json --tfidf-weight 0.3 --lda-weight 0.2 --embedding-weight 0.5

  # Save results to file
  python JobMain.py --resume resume.txt --jobs jobs.json --output matched_jobs.json

  # Show topic analysis
  python JobMain.py --jobs jobs.json --show-topics
        """
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to resume file (PDF, DOCX, or TXT)'
    )
    
    parser.add_argument(
        '--jobs',
        type=str,
        required=True,
        help='Path to jobs JSON file'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top matches to return (default: 20)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save matched jobs JSON (optional)'
    )
    
    parser.add_argument(
        '--tfidf-weight',
        type=float,
        default=0.25,
        help='Weight for TF-IDF scoring (default: 0.25)'
    )
    
    parser.add_argument(
        '--lda-weight',
        type=float,
        default=0.25,
        help='Weight for LDA topic scoring (default: 0.25)'
    )
    
    parser.add_argument(
        '--embedding-weight',
        type=float,
        default=0.50,
        help='Weight for embedding similarity (default: 0.50)'
    )
    
    parser.add_argument(
        '--n-topics',
        type=int,
        default=20,
        help='Number of topics for LDA model (default: 20)'
    )
    
    parser.add_argument(
        '--show-topics',
        action='store_true',
        help='Show discovered LDA topics and exit'
    )
    
    args = parser.parse_args()
    
    # Validate weights sum to 1.0
    total_weight = args.tfidf_weight + args.lda_weight + args.embedding_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Weights sum to {total_weight:.2f}, normalizing to 1.0")
        args.tfidf_weight /= total_weight
        args.lda_weight /= total_weight
        args.embedding_weight /= total_weight
    
    # Check if jobs file exists
    if not os.path.exists(args.jobs):
        print(f"Error: Jobs file not found: {args.jobs}")
        sys.exit(1)
    
    # Initialize job matcher
    print(f"Initializing Job Matcher...")
    print(f"Loading jobs from: {args.jobs}")
    matcher = JobMatcher(args.jobs, n_topics=args.n_topics)
    
    # Train models
    print("\nTraining NLP models (this may take a few minutes)...")
    matcher.train_models()
    
    # If only showing topics, display and exit
    if args.show_topics:
        matcher.print_topic_summary(n_words=10)
        return
    
    # Check if resume is provided
    if not args.resume:
        print("\nError: --resume argument is required for job matching")
        print("Use --show-topics flag to only view discovered topics")
        sys.exit(1)
    
    # Check if resume file exists
    if not os.path.exists(args.resume):
        print(f"Error: Resume file not found: {args.resume}")
        sys.exit(1)
    
    # Extract resume text
    print(f"\nExtracting text from resume: {args.resume}")
    try:
        extractor = ResumeExtractor()
        resume_text = extractor.extract_text(args.resume)
        print(f"Extracted {len(resume_text)} characters from resume")
    except Exception as e:
        print(f"Error extracting resume text: {str(e)}")
        sys.exit(1)
    
    # Prepare weights
    weights = {
        'tfidf': args.tfidf_weight,
        'lda': args.lda_weight,
        'embedding': args.embedding_weight
    }
    
    print(f"\nMatching weights:")
    print(f"  - TF-IDF (keyword matching): {weights['tfidf']:.2f}")
    print(f"  - LDA (topic similarity): {weights['lda']:.2f}")
    print(f"  - Embeddings (semantic similarity): {weights['embedding']:.2f}")
    
    # Match jobs
    print(f"\nFinding top {args.top} matching jobs...")
    results = matcher.match_jobs(resume_text, top_k=args.top, weights=weights)
    
    # Print results
    print_results(results, top_n=min(10, args.top))
    
    # Save results if output file specified
    if args.output:
        # Save JSON version
        save_results_to_json(results, args.output)
        
        # Also save readable text version with explanations
        text_output = args.output.replace('.json', '.txt')
        save_results_to_text(results, text_output, resume_file=args.resume, 
                           resume_text=resume_text, matcher=matcher)
        
        print(f"\nResults saved:")
        print(f"  - JSON format: {args.output}")
        print(f"  - Text format: {text_output}")
    else:
        # If no output specified, save with default name in results folder
        resume_name = os.path.splitext(os.path.basename(args.resume))[0]
        default_json = f"{resume_name}_matches.json"
        default_txt = f"{resume_name}_matches.txt"
        
        save_results_to_json(results, default_json)
        save_results_to_text(results, default_txt, resume_file=args.resume,
                           resume_text=resume_text, matcher=matcher)
        
        print(f"\nResults automatically saved:")
        print(f"  - JSON format: results/{default_json}")
        print(f"  - Text format: results/{default_txt}")
    
    # Summary statistics
    print("\nMatch Statistics:")
    print(f"  Average overall score: {sum(r['overall_score'] for r in results) / len(results):.4f}")
    print(f"  Highest score: {results[0]['overall_score']:.4f}")
    print(f"  Lowest score: {results[-1]['overall_score']:.4f}")
    
    print("\n[OK] Job matching complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
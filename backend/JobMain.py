#!/usr/bin/env python3
"""
Main Script for Job Matcher System
Easy-to-use interface for matching resumes to jobs using NLP techniques
"""

import argparse
import sys
import os
from JobMatcher import JobMatcher, print_results, save_results_to_json, save_results_to_text
from ResumeExtractor import ResumeExtractor
from ResumeParser import ResumeParser
from ResumeParser import ResumeParser


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
    
    parser.add_argument(
        '--parse-resume',
        action='store_true',
        help='Parse resume for structured info (years, degree, skills)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate resume against requirements'
    )
    
    parser.add_argument(
        '--min-years',
        type=int,
        help='Minimum years of experience required (for validation)'
    )
    
    parser.add_argument(
        '--required-degree',
        type=str,
        choices=['associates', 'bachelors', 'masters', 'phd'],
        help='Required education level (for validation)'
    )

    # NEW: Manual input for your experience and degree
    parser.add_argument(
        '--my-years',
        type=int,
        help='Your years of experience (overrides resume parsing)'
    )

    parser.add_argument(
        '--my-degree',
        type=str,
        choices=['associates', 'bachelors', 'masters', 'phd'],
        help='Your education level (overrides resume parsing)'
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
    
    # Parse resume if requested
    if args.parse_resume:
        print("\n" + "="*60)
        print("PARSING RESUME")
        print("="*60)
        parser_obj = ResumeParser()
        parsed_data = parser_obj.parse(resume_text)
        parser_obj.print_parsed_info(parsed_data)
    
    # Validate resume if requested
    if args.validate:
        print("\n" + "="*60)
        print("VALIDATING RESUME")
        print("="*60)
        
        # Build requirements dict
        requirements = {}
        if args.min_years:
            requirements['min_years'] = args.min_years
        if args.required_degree:
            requirements['required_degree'] = args.required_degree
        
        if not requirements:
            print("Warning: No validation requirements specified.")
            print("Use --min-years and/or --required-degree to set requirements.")
        else:
            parser_obj = ResumeParser()
            validation = parser_obj.validate_requirements(resume_text, requirements)
            
            print(f"\nRequirements:")
            if 'min_years' in requirements:
                print(f"  - Minimum years: {requirements['min_years']}")
            if 'required_degree' in requirements:
                print(f"  - Required degree: {requirements['required_degree']}")
            
            print(f"\nMeets Requirements: {'YES' if validation['meets_requirements'] else 'NO'}")
            
            for category, details in validation['details'].items():
                print(f"\n{category.replace('_', ' ').upper()}:")
                for key, value in details.items():
                    print(f"  {key}: {value}")
            
            if not validation['meets_requirements']:
                print("\n[!] Resume does not meet all requirements")
                print("You may still see job matches, but candidate may not qualify.")
            else:
                print("\n[OK] Resume meets all requirements!")
            
            print("="*60 + "\n")
    
    # Parse resume for structured information
    print("\nAnalyzing resume...")
    parser_resume = ResumeParser()
    parsed_info = parser_resume.parse(resume_text)

    # IMPROVED: Use manual input if provided, otherwise use parsed values
    if args.my_years is not None:
        resume_years = args.my_years
        print(f"\n[MANUAL INPUT] Using your specified years: {resume_years}")
    else:
        resume_years = parsed_info['years_of_experience']

    if args.my_degree is not None:
        resume_degree = args.my_degree
        print(f"[MANUAL INPUT] Using your specified degree: {resume_degree}")
    else:
        resume_degree = parsed_info['education_level']

    print("\nRESUME ANALYSIS:")
    print("-" * 60)
    if resume_years:
        print(f"  Years of Experience: {resume_years}")
    else:
        print(f"  Years of Experience: Not specified")

    if resume_degree:
        print(f"  Education Level: {resume_degree.title()}")
        if parsed_info['degree_details']:
            print(f"  Degree(s): {', '.join(parsed_info['degree_details'])}")
    else:
        print(f"  Education Level: Not specified")

    if parsed_info['skills']:
        print(f"  Skills: {len(parsed_info['skills'])} found")
        print(f"    Top skills: {', '.join(parsed_info['skills'][:5])}")

    print("-" * 60)

    # Store qualifications for later use
    resume_qualifications = {
        'years': resume_years,
        'degree': resume_degree
    }
    
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
    # IMPROVED: Pass resume years to matching algorithm for experience-based scoring
    results = matcher.match_jobs(
        resume_text,
        top_k=args.top,
        weights=weights,
        resume_years=resume_qualifications['years'],
        apply_experience_boost=True
    )
    
    # Always parse job requirements and add them to results
    print("Analyzing job requirements...")

    # IMPROVED: Track statistics about detected requirements
    jobs_with_years = 0
    jobs_with_degree = 0
    years_distribution = []

    for result in results:
        job = result['job']
        job_desc = job.get('description', '')

        # Parse job requirements
        job_reqs = parser_resume.parse_job_requirements(job_desc)
        result['job_requirements'] = job_reqs

        # IMPROVED: Track statistics
        if 'min_years' in job_reqs:
            jobs_with_years += 1
            years_distribution.append(job_reqs['min_years'])
        if 'required_degree' in job_reqs:
            jobs_with_degree += 1

        # Check if candidate qualifies (if we have their info)
        if resume_qualifications['years'] is not None or resume_qualifications['degree'] is not None:
            qualifies = parser_resume.candidate_qualifies(
                resume_qualifications['years'],
                resume_qualifications['degree'],
                job_reqs
            )
            result['candidate_qualifies'] = qualifies
        else:
            result['candidate_qualifies'] = None  # Unknown

    # IMPROVED: Display requirements detection statistics
    print(f"\nJOB REQUIREMENTS DETECTION:")
    print("-" * 60)
    print(f"  Jobs with years requirement detected: {jobs_with_years}/{len(results)}")
    print(f"  Jobs with degree requirement detected: {jobs_with_degree}/{len(results)}")
    if years_distribution:
        avg_years = sum(years_distribution) / len(years_distribution)
        min_years = min(years_distribution)
        max_years = max(years_distribution)
        print(f"  Years required: min={min_years}, max={max_years}, avg={avg_years:.1f}")
    else:
        print(f"  No explicit years requirements detected in job descriptions")
    print("-" * 60)
    
    # Filter jobs based on requirements if validation was requested
    if args.validate and (args.min_years or args.required_degree):
        print("\nFiltering jobs based on your qualifications...")
        
        filtered_results = []
        filtered_out = 0
        
        for result in results:
            if result.get('candidate_qualifies', True):  # Keep if qualifies or unknown
                filtered_results.append(result)
            else:
                filtered_out += 1
        
        if filtered_out > 0:
            print(f"[!] Filtered out {filtered_out} jobs where you don't meet minimum requirements")
            print(f"[OK] Showing {len(filtered_results)} jobs where you qualify\n")
        
        results = filtered_results
        
        if len(results) == 0:
            print("\n[!] No jobs found matching your qualifications.")
            print("Try:")
            print("  1. Removing --validate to see all matches")
            print("  2. Lowering requirements with --min-years or --required-degree")
            sys.exit(0)
    
    # Check if we have results
    if len(results) == 0:
        print("\n[!] No matching jobs found")
        sys.exit(0)
    
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
        
        # Different names for filtered vs unfiltered
        if args.validate and (args.min_years or args.required_degree):
            default_json = f"{resume_name}_matches_FILTERED.json"
            default_txt = f"{resume_name}_matches_FILTERED.txt"
        else:
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
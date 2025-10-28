#!/usr/bin/env python3
"""
Demo Script for Job Matcher
Tests the system with a sample resume
"""

from JobMatcher import JobMatcher, print_results, save_results_to_json
import os

# Sample resumes for different roles
SAMPLE_RESUMES = {
    "data_scientist": """
    JOHN DOE
    Senior Data Scientist
    john.doe@email.com | (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe
    
    PROFESSIONAL SUMMARY
    Results-driven Senior Data Scientist with 8+ years of experience in machine learning,
    deep learning, and statistical analysis. Expert in developing end-to-end ML solutions
    for business problems. Strong background in NLP, computer vision, and predictive modeling.
    
    TECHNICAL SKILLS
    • Languages: Python, R, SQL, Scala
    • ML Frameworks: TensorFlow, PyTorch, Keras, scikit-learn, XGBoost
    • Big Data: Apache Spark, Hadoop, Kafka
    • Cloud Platforms: AWS (SageMaker, EMR, S3), GCP (BigQuery, Vertex AI), Azure ML
    • Tools: Docker, Kubernetes, Git, MLflow, Airflow
    • Specialties: NLP, Computer Vision, Time Series Analysis, Recommender Systems
    
    PROFESSIONAL EXPERIENCE
    
    Senior Data Scientist | Tech Corp | 2020 - Present
    • Led development of recommendation engine serving 5M+ users, improving engagement by 35%
    • Built NLP pipeline for sentiment analysis on customer feedback (200K+ reviews daily)
    • Implemented computer vision models for product image classification (98% accuracy)
    • Mentored team of 4 junior data scientists
    • Technologies: Python, TensorFlow, BERT, AWS SageMaker
    
    Data Scientist | Analytics Inc | 2017 - 2020
    • Developed predictive models for customer churn (AUC: 0.92)
    • Created time series forecasting models for demand prediction
    • Built automated ML pipelines reducing model deployment time by 60%
    • Technologies: Python, scikit-learn, XGBoost, Apache Spark
    
    Junior Data Analyst | Data Solutions | 2015 - 2017
    • Performed statistical analysis on business metrics
    • Created interactive dashboards using Tableau and PowerBI
    • Conducted A/B tests for product features
    
    EDUCATION
    Ph.D. in Computer Science (Machine Learning) | Stanford University | 2015
    M.S. in Statistics | UC Berkeley | 2012
    B.S. in Mathematics | MIT | 2010
    
    PUBLICATIONS
    • "Deep Learning for Text Classification" - NeurIPS 2021
    • "Scalable Recommendation Systems" - KDD 2020
    • "Transfer Learning in Computer Vision" - CVPR 2019
    
    CERTIFICATIONS
    • AWS Certified Machine Learning - Specialty
    • TensorFlow Developer Certificate
    • Deep Learning Specialization (Coursera)
    """,
    
    "software_engineer": """
    JANE SMITH
    Senior Software Engineer
    jane.smith@email.com | (555) 987-6543 | GitHub: github.com/janesmith
    
    PROFESSIONAL SUMMARY
    Passionate Full-Stack Software Engineer with 6 years of experience building scalable
    web applications. Expert in modern JavaScript frameworks, cloud architecture, and
    microservices. Strong focus on code quality, testing, and agile development.
    
    TECHNICAL SKILLS
    • Languages: JavaScript/TypeScript, Python, Java, Go
    • Frontend: React, Vue.js, Angular, Next.js, HTML5, CSS3, Tailwind
    • Backend: Node.js, Express, Django, Spring Boot, FastAPI
    • Databases: PostgreSQL, MongoDB, Redis, MySQL, Elasticsearch
    • Cloud: AWS (EC2, Lambda, RDS, S3), GCP, Azure
    • DevOps: Docker, Kubernetes, Jenkins, CircleCI, Terraform
    • Testing: Jest, Pytest, Selenium, Cypress
    
    PROFESSIONAL EXPERIENCE
    
    Senior Software Engineer | WebTech Solutions | 2021 - Present
    • Architected microservices platform handling 10M+ requests/day
    • Led migration from monolith to microservices (reduced latency by 40%)
    • Implemented CI/CD pipelines automating deployment process
    • Mentored 3 junior developers
    • Stack: React, Node.js, TypeScript, PostgreSQL, AWS, Docker
    
    Software Engineer | StartupXYZ | 2019 - 2021
    • Built RESTful APIs serving mobile and web applications
    • Developed real-time chat feature using WebSockets
    • Optimized database queries (improved performance by 3x)
    • Stack: Vue.js, Express, MongoDB, Redis
    
    Junior Developer | Tech Agency | 2018 - 2019
    • Created responsive web applications for clients
    • Implemented payment integrations (Stripe, PayPal)
    • Fixed bugs and improved code quality
    
    EDUCATION
    B.S. in Computer Science | University of California, Berkeley | 2018
    
    PROJECTS
    • Open-source contributor to React ecosystem (500+ GitHub stars)
    • Created popular npm package for form validation (10K+ downloads/week)
    
    CERTIFICATIONS
    • AWS Certified Solutions Architect
    • Google Cloud Professional Developer
    """,
    
    "financial_analyst": """
    ROBERT JOHNSON
    Senior Financial Analyst
    robert.johnson@email.com | (555) 456-7890
    
    PROFESSIONAL SUMMARY
    Detail-oriented Senior Financial Analyst with 7 years of experience in financial planning,
    analysis, and reporting. Expert in financial modeling, budgeting, and forecasting.
    Strong analytical skills with proven track record of driving business insights.
    
    TECHNICAL SKILLS
    • Financial Modeling & Valuation (DCF, Comparable Analysis, LBO)
    • Financial Planning & Analysis (FP&A)
    • Budgeting & Forecasting
    • Tools: Excel (Advanced), SQL, Python, Tableau, PowerBI
    • ERP Systems: SAP, Oracle, NetSuite
    • Financial Software: Bloomberg, FactSet, Capital IQ
    
    PROFESSIONAL EXPERIENCE
    
    Senior Financial Analyst | Fortune 500 Corp | 2020 - Present
    • Lead quarterly forecasting and annual budgeting process ($500M+ budget)
    • Built financial models for strategic initiatives and M&A analysis
    • Developed executive dashboards for C-suite decision-making
    • Identified cost savings opportunities ($15M annually)
    • Presented financial insights to board of directors
    
    Financial Analyst | Investment Bank | 2018 - 2020
    • Performed valuation analysis for client companies
    • Created detailed financial models for pitch books
    • Conducted industry and competitive analysis
    • Supported due diligence for M&A transactions
    
    Associate Analyst | Consulting Firm | 2016 - 2018
    • Analyzed financial statements and operational metrics
    • Prepared client presentations and reports
    • Performed variance analysis and identified trends
    
    EDUCATION
    MBA (Finance) | Harvard Business School | 2016
    B.S. in Finance | University of Pennsylvania (Wharton) | 2014
    
    CERTIFICATIONS
    • Chartered Financial Analyst (CFA)
    • Certified Public Accountant (CPA)
    • Financial Modeling & Valuation Analyst (FMVA)
    
    ACHIEVEMENTS
    • Developed financial model that identified $20M cost reduction opportunity
    • Improved forecast accuracy by 25% through enhanced modeling techniques
    • Led successful integration of acquired company ($100M deal)
    """
}


def run_demo(jobs_file: str, resume_type: str = "data_scientist"):
    """
    Run a demo of the job matcher
    
    Args:
        jobs_file: Path to jobs JSON file
        resume_type: Type of sample resume to use
    """
    
    print("="*80)
    print("JOB MATCHER DEMO")
    print("="*80)
    
    # Check if jobs file exists
    if not os.path.exists(jobs_file):
        print(f"\nError: Jobs file not found: {jobs_file}")
        print("Please ensure job_results.json is in the current directory")
        return
    
    # Get sample resume
    if resume_type not in SAMPLE_RESUMES:
        print(f"\nError: Unknown resume type: {resume_type}")
        print(f"Available types: {', '.join(SAMPLE_RESUMES.keys())}")
        return
    
    resume_text = SAMPLE_RESUMES[resume_type]
    
    print(f"\nUsing sample resume: {resume_type.replace('_', ' ').title()}")
    print(f"Resume length: {len(resume_text)} characters")
    
    # Initialize matcher
    print(f"\nInitializing Job Matcher...")
    matcher = JobMatcher(jobs_file, n_topics=20)
    
    # Train models
    print("\nTraining NLP models...")
    matcher.train_models()
    
    # Show topics
    print("\nDiscovered Topics in Job Market:")
    matcher.print_topic_summary(n_words=8)
    
    # Match jobs with default weights
    print(f"\nMatching jobs (default weights)...")
    results_default = matcher.match_jobs(resume_text, top_k=20)
    
    print("\nRESULTS WITH DEFAULT WEIGHTS (25% TF-IDF, 25% LDA, 50% Embeddings):")
    print_results(results_default, top_n=5)
    
    # Match jobs with keyword-heavy weights
    print("\nMatching jobs (keyword-heavy weights)...")
    results_keywords = matcher.match_jobs(
        resume_text,
        top_k=20,
        weights={'tfidf': 0.5, 'lda': 0.2, 'embedding': 0.3}
    )
    
    print("\nRESULTS WITH KEYWORD-HEAVY WEIGHTS (50% TF-IDF, 20% LDA, 30% Embeddings):")
    print_results(results_keywords, top_n=5)
    
    # Match jobs with semantic-heavy weights
    print("\nMatching jobs (semantic-heavy weights)...")
    results_semantic = matcher.match_jobs(
        resume_text,
        top_k=20,
        weights={'tfidf': 0.2, 'lda': 0.2, 'embedding': 0.6}
    )
    
    print("\nRESULTS WITH SEMANTIC-HEAVY WEIGHTS (20% TF-IDF, 20% LDA, 60% Embeddings):")
    print_results(results_semantic, top_n=5)
    
    # Save results
    output_file = f"demo_results_{resume_type}.json"
    save_results_to_json(results_default, output_file)
    
    # Comparison
    print("\nCOMPARISON:")
    print(f"  Default weights avg score: {sum(r['overall_score'] for r in results_default) / len(results_default):.4f}")
    print(f"  Keyword-heavy avg score: {sum(r['overall_score'] for r in results_keywords) / len(results_keywords):.4f}")
    print(f"  Semantic-heavy avg score: {sum(r['overall_score'] for r in results_semantic) / len(results_semantic):.4f}")
    
    print(f"\n✓ Demo complete! Results saved to {output_file}")
    print("\n" + "="*80)


def main():
    """Main function"""
    import sys
    
    jobs_file = "job_results.json"
    resume_type = "data_scientist"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        jobs_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        resume_type = sys.argv[2]
    
    # Run demo
    run_demo(jobs_file, resume_type)
    
    # Show available resume types
    print("\nAvailable sample resumes:")
    for key in SAMPLE_RESUMES.keys():
        print(f"  - {key}")
    print(f"\nUsage: python demo.py [jobs_file] [resume_type]")
    print(f"Example: python demo.py job_results.json software_engineer")


if __name__ == "__main__":
    main()

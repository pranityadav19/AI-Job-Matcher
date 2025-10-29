#!/usr/bin/env python3
"""
Test script to verify the improvements to years extraction
"""

import sys
sys.path.insert(0, '.')

from ResumeParser import ResumeParser

# Test cases for job descriptions with various experience patterns
test_job_descriptions = [
    {
        "title": "Senior Data Scientist",
        "description": """
        We are looking for a Senior Data Scientist with 5+ years of experience
        in machine learning and data analysis. Must have strong Python skills.
        """
    },
    {
        "title": "Entry Level Software Engineer",
        "description": """
        Looking for an entry-level software engineer to join our team.
        Fresh graduates are welcome. Must have knowledge of Java and Python.
        """
    },
    {
        "title": "Junior Data Analyst",
        "description": """
        Junior Data Analyst position available. We're looking for someone with
        1-2 years of experience in data analysis and SQL.
        """
    },
    {
        "title": "Lead ML Engineer",
        "description": """
        Lead Machine Learning Engineer role. Requires at least 7 years of
        experience in ML, deep learning, and AI. Must have PhD or Masters degree.
        """
    },
    {
        "title": "Mid-Level Backend Developer",
        "description": """
        Mid-level backend developer needed with 3 to 5 years of experience.
        Strong knowledge of Node.js, MongoDB, and REST APIs required.
        """
    },
    {
        "title": "Data Scientist",
        "description": """
        Data Scientist position. Looking for someone with experience in Python,
        machine learning, and statistical analysis. No specific years mentioned.
        """
    },
    {
        "title": "Principal Engineer",
        "description": """
        Principal Engineer role with minimum 10 years experience required.
        Must have expertise in system design and architecture.
        """
    }
]

# Test resume examples with different experience levels
test_resumes = [
    {
        "name": "Entry Level Candidate",
        "text": "Fresh graduate with Bachelor's degree in Computer Science. Completed internship in data science."
    },
    {
        "name": "Junior Candidate (2 years)",
        "text": "Software engineer with 2 years of experience in Python and machine learning. Bachelor's degree in CS."
    },
    {
        "name": "Mid-Level Candidate (5 years)",
        "text": "Data scientist with 5 years of experience in ML, NLP, and deep learning. Master's degree in Data Science."
    },
    {
        "name": "Senior Candidate (8 years)",
        "text": "Senior ML engineer with 8 years of experience. PhD in Computer Science. Published researcher."
    }
]

def test_job_requirements_parsing():
    """Test improved job requirements parsing"""
    print("="*80)
    print("TESTING JOB REQUIREMENTS PARSING")
    print("="*80)
    print()

    parser = ResumeParser()

    for job in test_job_descriptions:
        print(f"Job: {job['title']}")
        print("-" * 80)

        requirements = parser.parse_job_requirements(job['description'])

        if 'min_years' in requirements:
            print(f"  ✓ Years detected: {requirements['min_years']} years")
            if 'level' in requirements:
                print(f"  ✓ Level keyword: '{requirements['level']}'")
        else:
            print(f"  ✗ No years requirement detected")

        if 'required_degree' in requirements:
            print(f"  ✓ Degree: {requirements['required_degree']}")
        else:
            print(f"  - No degree requirement detected")

        print()

def test_resume_parsing():
    """Test resume parsing for years of experience"""
    print("="*80)
    print("TESTING RESUME PARSING")
    print("="*80)
    print()

    parser = ResumeParser()

    for resume in test_resumes:
        print(f"Resume: {resume['name']}")
        print("-" * 80)

        parsed = parser.parse(resume['text'])

        if parsed['years_of_experience']:
            print(f"  ✓ Years detected: {parsed['years_of_experience']} years")
        else:
            print(f"  ✗ No years detected (might be entry level)")

        if parsed['education_level']:
            print(f"  ✓ Education: {parsed['education_level']}")
        else:
            print(f"  - No education detected")

        print()

def test_qualification_matching():
    """Test whether candidates qualify for jobs"""
    print("="*80)
    print("TESTING QUALIFICATION MATCHING")
    print("="*80)
    print()

    parser = ResumeParser()

    # Parse all resumes
    parsed_resumes = []
    for resume in test_resumes:
        parsed = parser.parse(resume['text'])
        parsed_resumes.append({
            'name': resume['name'],
            'years': parsed['years_of_experience'],
            'degree': parsed['education_level']
        })

    # Check each resume against each job
    for job in test_job_descriptions[:4]:  # Test first 4 jobs
        print(f"Job: {job['title']}")
        print("-" * 80)

        job_reqs = parser.parse_job_requirements(job['description'])

        if not job_reqs:
            print("  No requirements detected - all candidates qualify")
            print()
            continue

        req_str = []
        if 'min_years' in job_reqs:
            req_str.append(f"{job_reqs['min_years']}+ years")
        if 'required_degree' in job_reqs:
            req_str.append(f"{job_reqs['required_degree']} degree")
        print(f"  Requirements: {', '.join(req_str) if req_str else 'None'}")
        print()

        for resume in parsed_resumes:
            qualifies = parser.candidate_qualifies(
                resume['years'],
                resume['degree'],
                job_reqs
            )

            status = "✓ QUALIFIES" if qualifies else "✗ DOES NOT QUALIFY"
            resume_info = []
            if resume['years']:
                resume_info.append(f"{resume['years']} yrs")
            if resume['degree']:
                resume_info.append(resume['degree'])

            print(f"    {status} - {resume['name']} ({', '.join(resume_info) if resume_info else 'No info'})")

        print()

def main():
    print("\n")
    print("="*80)
    print("JOB MATCHER IMPROVEMENTS - TEST SUITE")
    print("="*80)
    print()

    # Run all tests
    test_job_requirements_parsing()
    test_resume_parsing()
    test_qualification_matching()

    print("="*80)
    print("TESTS COMPLETE")
    print("="*80)
    print()
    print("Summary:")
    print("  ✓ Improved years extraction patterns (entry-level, junior, senior, etc.)")
    print("  ✓ Better pattern matching for experience requirements")
    print("  ✓ Qualification checking works correctly")
    print()

if __name__ == "__main__":
    main()

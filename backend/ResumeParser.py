"""
Resume Parser
Extracts structured information from resume text:
- Years of experience
- Education level (degree)
- Skills
- Job titles
"""

import re
from typing import Dict, List, Optional


class ResumeParser:
    """Parse structured information from resume text"""
    
    def __init__(self):
        """Initialize parser with common patterns and keywords"""
        
        # Degree keywords
        self.degrees = {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'masters': ['masters', 'master', 'ms', 'm.s', 'mba', 'm.b.a', 'ma', 'm.a'],
            'bachelors': ['bachelors', 'bachelor', 'bs', 'b.s', 'ba', 'b.a', 'bsc', 'b.sc'],
            'associates': ['associates', 'associate', 'as', 'a.s', 'aa', 'a.a']
        }
        
        # Common skills (can be expanded)
        self.tech_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'cplusplus', 'csharp', 'c#',
            'react', 'angular', 'vue', 'nodejs', 'node.js',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'scikit learn',
            'git', 'github', 'gitlab', 'jenkins', 'ci/cd',
            'html', 'css', 'tailwind', 'bootstrap',
            'rest', 'api', 'graphql', 'microservices',
            'agile', 'scrum', 'jira'
        ]
        
        # Job title keywords
        self.job_titles = [
            'data scientist', 'machine learning engineer', 'ml engineer',
            'software engineer', 'software developer', 'full stack', 'backend', 'frontend',
            'data analyst', 'business analyst', 'data engineer',
            'devops engineer', 'cloud engineer', 'solutions architect',
            'product manager', 'project manager', 'program manager',
            'qa engineer', 'test engineer', 'sre'
        ]
    
    def parse(self, resume_text: str) -> Dict:
        """
        Parse resume and extract structured information
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Dictionary with parsed information
        """
        text_lower = resume_text.lower()
        
        return {
            'years_of_experience': self.extract_years_of_experience(text_lower),
            'education_level': self.extract_education_level(text_lower),
            'degree_details': self.extract_degree_details(resume_text),
            'skills': self.extract_skills(text_lower),
            'job_titles': self.extract_job_titles(text_lower),
            'has_phd': self.has_degree_level(text_lower, 'phd'),
            'has_masters': self.has_degree_level(text_lower, 'masters'),
            'has_bachelors': self.has_degree_level(text_lower, 'bachelors')
        }
    
    def extract_years_of_experience(self, text: str) -> Optional[int]:
        """
        Extract years of experience from resume
        
        Looks for patterns like:
        - "5 years of experience"
        - "5+ years"
        - "5-7 years"
        """
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\+?\s*yrs?\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\+?\s*years?',
            r'(\d+)-\d+\s*years?\s+(?:of\s+)?experience',
            r'over\s+(\d+)\s+years?',
            r'more than\s+(\d+)\s+years?'
        ]
        
        years_found = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    if 0 < years <= 50:  # Sanity check
                        years_found.append(years)
                except ValueError:
                    continue
        
        if years_found:
            # Return the maximum (most conservative estimate)
            return max(years_found)
        
        return None
    
    def extract_education_level(self, text: str) -> Optional[str]:
        """
        Extract highest education level
        
        Returns: 'phd', 'masters', 'bachelors', 'associates', or None
        """
        # Check in order of highest to lowest
        for level, keywords in self.degrees.items():
            for keyword in keywords:
                # Look for keyword with word boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        
        return None
    
    def has_degree_level(self, text: str, level: str) -> bool:
        """Check if resume has a specific degree level"""
        if level not in self.degrees:
            return False
        
        for keyword in self.degrees[level]:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def extract_degree_details(self, text: str) -> List[str]:
        """
        Extract degree details (field of study)
        
        Looks for patterns like:
        - "Bachelor of Science in Computer Science"
        - "MS in Data Science"
        - "PhD in Machine Learning"
        """
        degree_patterns = [
            r'(?:phd|ph\.d|doctorate|doctoral)\s+in\s+([a-zA-Z\s]+)',
            r'(?:masters?|ms|m\.s|mba|m\.b\.a|ma|m\.a)\s+in\s+([a-zA-Z\s]+)',
            r'(?:bachelors?|bachelor|bs|b\.s|ba|b\.a)\s+in\s+([a-zA-Z\s]+)',
            r'(?:bachelors?|bachelor)\s+of\s+(?:science|arts)\s+in\s+([a-zA-Z\s]+)',
            r'(?:masters?|master)\s+of\s+(?:science|arts|business)\s+in\s+([a-zA-Z\s]+)'
        ]
        
        degrees = []
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the field of study
                field = match.strip()
                # Remove trailing words like "from", "at"
                field = re.sub(r'\s+(?:from|at|university|college).*$', '', field, flags=re.IGNORECASE)
                if field and len(field) > 3:
                    degrees.append(field.title())
        
        return degrees
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from resume"""
        found_skills = []
        
        for skill in self.tech_skills:
            # Use word boundaries for exact matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.append(skill.title())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in found_skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills
    
    def extract_job_titles(self, text: str) -> List[str]:
        """Extract job titles from resume"""
        found_titles = []
        
        for title in self.job_titles:
            pattern = r'\b' + re.escape(title) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_titles.append(title.title())
        
        # Remove duplicates
        return list(set(found_titles))
    
    def validate_requirements(self, resume_text: str, requirements: Dict) -> Dict:
        """
        Validate if resume meets certain requirements
        
        Args:
            resume_text: Resume text
            requirements: Dict like {
                'min_years': 3,
                'required_degree': 'bachelors',  # or 'masters', 'phd'
                'required_skills': ['python', 'machine learning']
            }
            
        Returns:
            Dict with validation results
        """
        parsed = self.parse(resume_text)
        
        results = {
            'meets_requirements': True,
            'details': {}
        }
        
        # Check years of experience
        if 'min_years' in requirements:
            min_years = requirements['min_years']
            candidate_years = parsed['years_of_experience']
            
            if candidate_years is None:
                results['meets_requirements'] = False
                results['details']['years_of_experience'] = {
                    'required': min_years,
                    'found': 'Not specified',
                    'meets': False
                }
            elif candidate_years < min_years:
                results['meets_requirements'] = False
                results['details']['years_of_experience'] = {
                    'required': min_years,
                    'found': candidate_years,
                    'meets': False
                }
            else:
                results['details']['years_of_experience'] = {
                    'required': min_years,
                    'found': candidate_years,
                    'meets': True
                }
        
        # Check education level
        if 'required_degree' in requirements:
            required = requirements['required_degree']
            candidate_degree = parsed['education_level']
            
            # Degree hierarchy
            degree_levels = {
                'associates': 1,
                'bachelors': 2,
                'masters': 3,
                'phd': 4
            }
            
            if candidate_degree is None:
                results['meets_requirements'] = False
                results['details']['education'] = {
                    'required': required,
                    'found': 'Not specified',
                    'meets': False
                }
            else:
                req_level = degree_levels.get(required, 0)
                cand_level = degree_levels.get(candidate_degree, 0)
                
                meets = cand_level >= req_level
                if not meets:
                    results['meets_requirements'] = False
                
                results['details']['education'] = {
                    'required': required,
                    'found': candidate_degree,
                    'meets': meets
                }
        
        # Check required skills
        if 'required_skills' in requirements:
            required_skills = [s.lower() for s in requirements['required_skills']]
            candidate_skills = [s.lower() for s in parsed['skills']]
            
            missing_skills = [s for s in required_skills if s not in candidate_skills]
            
            if missing_skills:
                results['meets_requirements'] = False
                results['details']['skills'] = {
                    'required': requirements['required_skills'],
                    'found': parsed['skills'],
                    'missing': missing_skills,
                    'meets': False
                }
            else:
                results['details']['skills'] = {
                    'required': requirements['required_skills'],
                    'found': parsed['skills'],
                    'missing': [],
                    'meets': True
                }
        
        return results
    
    def print_parsed_info(self, parsed_data: Dict):
        """Pretty print parsed resume information"""
        print("\n" + "="*60)
        print("PARSED RESUME INFORMATION")
        print("="*60 + "\n")
        
        print(f"Years of Experience: {parsed_data['years_of_experience'] or 'Not found'}")
        print(f"Education Level: {parsed_data['education_level'] or 'Not found'}")
        
        if parsed_data['degree_details']:
            print(f"\nDegree Details:")
            for degree in parsed_data['degree_details']:
                print(f"  - {degree}")
        
        if parsed_data['skills']:
            print(f"\nSkills Found ({len(parsed_data['skills'])}):")
            for skill in parsed_data['skills'][:20]:  # Show first 20
                print(f"  - {skill}")
            if len(parsed_data['skills']) > 20:
                print(f"  ... and {len(parsed_data['skills']) - 20} more")
        
        if parsed_data['job_titles']:
            print(f"\nJob Titles Found:")
            for title in parsed_data['job_titles']:
                print(f"  - {title}")
        
        print("\n" + "="*60 + "\n")
    
    def parse_job_requirements(self, job_description: str) -> Dict:
        """
        Parse job requirements from job description
        
        Args:
            job_description: Job description text
            
        Returns:
            Dict with requirements like {'min_years': 3, 'required_degree': 'bachelors'}
        """
        text_lower = job_description.lower()
        requirements = {}
        
        # Extract minimum years required
        years_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)',
            r'minimum\s+(?:of\s+)?(\d+)\s*years?',
            r'at least\s+(\d+)\s*years?',
            r'(\d+)\+\s*years?'
        ]
        
        years_found = []
        for pattern in years_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    if 0 < years <= 20:  # Sanity check
                        years_found.append(years)
                except ValueError:
                    continue
        
        if years_found:
            requirements['min_years'] = min(years_found)  # Use minimum to be inclusive
        
        # Extract degree requirements
        degree_requirements = []
        
        if re.search(r'\b(?:phd|ph\.d|doctorate)\b', text_lower):
            degree_requirements.append('phd')
        if re.search(r'\b(?:masters?|ms|m\.s|mba)\b', text_lower):
            degree_requirements.append('masters')
        if re.search(r'\b(?:bachelors?|bachelor|bs|b\.s|ba|b\.a|undergraduate)\b', text_lower):
            degree_requirements.append('bachelors')
        
        # Use the highest degree mentioned as requirement
        degree_hierarchy = {'bachelors': 2, 'masters': 3, 'phd': 4}
        if degree_requirements:
            highest = max(degree_requirements, key=lambda d: degree_hierarchy.get(d, 0))
            requirements['required_degree'] = highest
        
        return requirements
    
    def candidate_qualifies(self, resume_years: Optional[int], resume_degree: Optional[str], 
                           job_requirements: Dict) -> bool:
        """
        Check if candidate qualifies for a job based on requirements
        
        Args:
            resume_years: Years of experience from resume
            resume_degree: Degree level from resume
            job_requirements: Dict with 'min_years' and/or 'required_degree'
            
        Returns:
            True if candidate meets or exceeds requirements
        """
        # Degree hierarchy
        degree_levels = {
            'associates': 1,
            'bachelors': 2,
            'masters': 3,
            'phd': 4
        }
        
        # Check years
        if 'min_years' in job_requirements:
            if resume_years is None:
                return False  # Can't verify, so fail
            if resume_years < job_requirements['min_years']:
                return False
        
        # Check degree
        if 'required_degree' in job_requirements:
            if resume_degree is None:
                return False  # Can't verify, so fail
            
            req_level = degree_levels.get(job_requirements['required_degree'], 0)
            cand_level = degree_levels.get(resume_degree, 0)
            
            if cand_level < req_level:
                return False
        
        return True


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ResumeParser.py <resume_file>")
        print("\nExample:")
        print("  python ResumeParser.py my_resume.pdf")
        sys.exit(1)
    
    from ResumeExtractor import ResumeExtractor
    
    resume_file = sys.argv[1]
    
    # Extract text
    print(f"Extracting text from {resume_file}...")
    extractor = ResumeExtractor()
    resume_text = extractor.extract_text(resume_file)
    
    # Parse resume
    print("Parsing resume...")
    parser = ResumeParser()
    parsed = parser.parse(resume_text)
    
    # Display results
    parser.print_parsed_info(parsed)
    
    # Example validation
    print("\nEXAMPLE VALIDATION:")
    print("-"*60)
    requirements = {
        'min_years': 3,
        'required_degree': 'bachelors',
        'required_skills': ['python', 'machine learning']
    }
    
    validation = parser.validate_requirements(resume_text, requirements)
    
    print(f"\nRequirements:")
    print(f"  - Min years: {requirements['min_years']}")
    print(f"  - Degree: {requirements['required_degree']}")
    print(f"  - Skills: {', '.join(requirements['required_skills'])}")
    
    print(f"\nMeets Requirements: {validation['meets_requirements']}")
    
    for category, details in validation['details'].items():
        print(f"\n{category.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
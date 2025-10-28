# AI Job Matcher

Intelligent job matching system using NLP to match resumes with job postings. Analyzes qualifications and shows WHY each job matches.

## Features

- **Multi-Method Matching**: TF-IDF, LDA Topic Modeling, Sentence Embeddings
- **Match Explanations**: Shows WHY each job matched with keywords and overlap %
- **Resume Parsing**: Extracts years of experience, education level, degree details, and skills
- **Qualification Filtering**: Filter jobs based on your experience and education requirements
- **Job Requirements Analysis**: Automatically detects job requirements and shows if you qualify
- **Multiple Formats**: Supports PDF, DOCX, TXT resumes
- **Smart Results**: Auto-saves to JSON and readable text files with detailed breakdowns

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Run
python JobMain.py --resume your_resume.pdf --jobs job_results.json
```

## Results

Outputs saved to `results/` folder:

- **TXT file**: Human-readable with TOP 3 summary, match explanations, and qualification status for EVERY job
- **JSON file**: Machine-readable data
- **Filtered files**: When using `--validate`, saves as `*_FILTERED.txt/json`

## Usage Examples

```bash
# Basic matching
python JobMain.py --resume resume.pdf --jobs jobs.json

# Parse your resume (shows years, degree, skills)
python JobMain.py --resume resume.pdf --jobs jobs.json --parse-resume

# Filter by qualifications (only jobs you qualify for)
python JobMain.py --resume resume.pdf --jobs jobs.json --validate --min-years 3 --required-degree bachelors

# Get more matches
python JobMain.py --resume resume.pdf --jobs jobs.json --top 50

# Custom weights
python JobMain.py --resume resume.pdf --jobs jobs.json --tfidf-weight 0.5 --lda-weight 0.2 --embedding-weight 0.3

# Run demo
python demo.py

# Parse resume only
python ResumeParser.py your_resume.pdf
```

## Key Options

```
--resume PATH              Path to your resume (PDF, DOCX, or TXT)
--jobs PATH                Path to jobs JSON file
--top N                    Number of matches (default: 20)
--parse-resume             Show extracted resume info
--validate                 Filter jobs by qualifications
--min-years N              Minimum years required
--required-degree LEVEL    Required degree (associates/bachelors/masters/phd)
--show-topics              Display job market topics
```

## How Qualification Filtering Works

The system automatically:

1. Parses YOUR resume for years of experience and degree
2. Parses EACH job description for requirements
3. Compares them intelligently (e.g., Masters qualifies for Bachelor requirements)
4. Shows qualification status for every job
5. Optionally filters out jobs you don't qualify for (with `--validate`)

**Every job shows:**

- Requirements detected (years + degree)
- Your status: `[YOU QUALIFY]` or `[DOES NOT MEET REQUIREMENTS]`

## Tech Stack

- Python 3.12
- NLTK, scikit-learn, sentence-transformers
- PyPDF2, python-docx

## Next Steps

- [x] Resume parsing (years, degree, skills)
- [x] Job requirements analysis
- [x] Qualification filtering
- [ ] Web frontend (FastAPI)
- [ ] Security and Privacy
- [ ] RAG Pipeline Database with Scheduled Updating
- [ ] Wider selection in Database
- [ ] Mobile App
- [ ] Model persistence
- [ ] Enhanced skill extraction
- [ ] User accounts

## Documentation

See `QUICK_START_GUIDE.txt` for detailed instructions.

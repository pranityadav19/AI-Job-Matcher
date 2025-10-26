# AI Job Matcher

Intelligent job matching system using NLP to match resumes with job postings.

## Features

- **Multi-Method Matching**: TF-IDF, LDA Topic Modeling, Sentence Embeddings
- **Match Explanations**: Shows WHY each job matched with keywords and overlap %
- **Multiple Formats**: Supports PDF, DOCX, TXT resumes
- **Smart Results**: Auto-saves to JSON and readable text files

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
- **TXT file**: Human-readable with TOP 3 summary and explanations
- **JSON file**: Machine-readable data

## Usage Examples

```bash
# Basic
python JobMain.py --resume resume.pdf --jobs jobs.json

# Custom matches
python JobMain.py --resume resume.pdf --jobs jobs.json --top 50

# Run demo
python demo.py
```

## Tech Stack

- Python 3.12
- NLTK, scikit-learn, sentence-transformers
- PyPDF2, python-docx

## Next Steps

- [ ] Web frontend (Flask/FastAPI)
- [ ] Model persistence
- [ ] Skill extraction
- [ ] User accounts

## Documentation

See `QUICK_START_GUIDE.txt` for detailed instructions.

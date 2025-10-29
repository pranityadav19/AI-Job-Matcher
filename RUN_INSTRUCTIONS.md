# How to Run Your Improved Job Matcher

## Quick Setup (First Time Only)

### Step 1: Install Dependencies

```bash
# Navigate to the backend folder
cd backend

# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

This will take a few minutes to download all the ML models.

---

## Running the Job Matcher

### Basic Run (Recommended)

```bash
cd backend
python JobMain.py --resume Pranit_Yadav_Resume.pdf --jobs job_results.json --top 20
```

This will:
- ✓ Parse your resume for years of experience, education, and skills
- ✓ Apply the NEW experience-based scoring algorithm
- ✓ Show statistics on detected job requirements
- ✓ Match you with top 20 jobs
- ✓ Save results to `results/` folder

### See Detailed Resume Analysis

```bash
python JobMain.py --resume Pranit_Yadav_Resume.pdf --jobs job_results.json --parse-resume
```

This shows what was extracted from your resume:
- Years of experience
- Education level and degree details
- Skills detected

### Filter to Only Jobs You Qualify For

```bash
python JobMain.py --resume Pranit_Yadav_Resume.pdf --jobs job_results.json --validate --min-years 2
```

This will:
- Only show jobs where you meet the years requirement
- Filter out jobs that require more experience than you have
- Save filtered results as `*_FILTERED.txt` and `*_FILTERED.json`

### Get More Matches

```bash
python JobMain.py --resume Pranit_Yadav_Resume.pdf --jobs job_results.json --top 50
```

Shows top 50 matches instead of default 20.

---

## Understanding the Output

### Console Output

You'll see:
```
RESUME ANALYSIS:
  Years of Experience: 5
  Education Level: Bachelors
  Skills Detected: 10

Matching weights:
  - TF-IDF (keyword matching): 0.25
  - LDA (topic similarity): 0.25
  - Embeddings (semantic similarity): 0.50

Finding top 20 matching jobs...
Applying experience-based score adjustment (resume: 5 years)...  ← NEW!

JOB REQUIREMENTS DETECTION:                                      ← NEW!
  Jobs with years requirement detected: 15/20                    ← NEW!
  Jobs with degree requirement detected: 8/20                    ← NEW!
  Years required: min=0, max=8, avg=3.5                         ← NEW!
```

### Results Files (in `results/` folder)

**Text File** (`Pranit_Yadav_Resume_matches.txt`):
- TOP 3 SUMMARY with quick overview
- Full details for ALL matched jobs including:
  - Match scores (overall + breakdown)
  - Job requirements detected
  - Your qualification status: `[YOU QUALIFY]` or `[DOES NOT MEET REQUIREMENTS]`
  - WHY the job matched (keywords, overlap %, match strength)

**JSON File** (`Pranit_Yadav_Resume_matches.json`):
- Machine-readable format for further analysis

---

## What's New (Thanks to Improvements!)

### 1. Experience-Based Ranking
Jobs are now automatically ranked higher/lower based on how well your experience matches:
- **Perfect match** (your years = required): +10% score boost
- **Slightly more experience**: +8% boost
- **Much more experience**: May be ranked lower (you might be overqualified)
- **Less experience**: Ranked lower (underqualified)

### 2. Better Detection
Now detects years even when written as:
- "Senior" (assumes 5+ years)
- "Entry-level" (assumes 0 years)
- "Junior" (assumes 1+ years)
- "Mid-level" (assumes 3+ years)
- "3-5 years", "between 3 and 5 years", etc.

### 3. More Visibility
You can now see:
- How many jobs have years requirements
- Average years required across all matches
- Whether you qualify for each job

---

## Testing the Improvements

Want to see if the improvements work? Run the test script:

```bash
cd backend
python test_improvements.py
```

This will show you:
- How well the parser detects years in various job descriptions
- How qualification matching works
- Test results with example resumes and jobs

---

## Troubleshooting

### Error: "No module named 'sentence_transformers'"
Run: `pip install -r requirements.txt`

### Error: "Jobs file not found"
Make sure you're in the `backend` folder and `job_results.json` exists there.

### Taking too long?
The first run trains ML models (takes 2-5 minutes). Subsequent runs are faster.

### Not detecting my years?
Make sure your resume explicitly says "5 years of experience" or similar.
The parser looks for patterns like:
- "5 years of experience"
- "5+ years"
- "Over 5 years"

---

## Example Commands

```bash
# Full analysis with filtering
python JobMain.py --resume Pranit_Yadav_Resume.pdf --jobs job_results.json --parse-resume --validate --min-years 2 --top 30

# Just get top matches (fast)
python JobMain.py --resume Pranit_Yadav_Resume.pdf --jobs job_results.json

# See what topics the job market has
python JobMain.py --jobs job_results.json --show-topics
```

---

## Need Help?

Check the full README: `backend/README.md`

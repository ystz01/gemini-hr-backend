import os
import json
import uuid
import shutil
import tempfile
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

import firebase_admin
from firebase_admin import credentials, firestore, storage

# =============================================================
# 1. APPLICATION & CORS SETUP
# =============================================================
app = FastAPI(title="Gemini HR Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# 2. FIREBASE & GEMINI CLIENT INITIALIZATION
# =============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
FIREBASE_STORAGE_BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET", "")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    if FIREBASE_SERVICE_ACCOUNT:
        if os.path.exists(FIREBASE_SERVICE_ACCOUNT):
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT)
        else:
            cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT)
            cred = credentials.Certificate(cred_dict)
    elif os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
    else:
        cred = None

    if cred:
        firebase_admin.initialize_app(cred, {
            "storageBucket": FIREBASE_STORAGE_BUCKET
        })

db = firestore.client() if firebase_admin._apps else None
bucket = storage.bucket() if firebase_admin._apps else None

# =============================================================
# 3. PYDANTIC SCHEMAS
# =============================================================
class WorkExperience(BaseModel):
    role: str = Field(description="Job title or designation")
    company: str = Field(description="Company or organization name")
    duration: str = Field(description="Dates or duration of employment")

class ProjectInfo(BaseModel):
    title: str = Field(description="Project title or name")
    description: str = Field(description="Brief overview, technologies used, or outcome of the project")

class CandidateProfile(BaseModel):
    name: str = Field(description="Full name of candidate")
    email: Optional[str] = Field(default="", description="Email address")
    phone: Optional[str] = Field(default="", description="Phone number")
    location: str = Field(description="City or State, e.g. Johor Bahru, Penang")
    highest_education: str = Field(description="Degree and university name")
    education_level: str = Field(description="Normalized: Diploma, Bachelor, Master, PhD, or High School")
    skills: List[str] = Field(description="List of technical and hard skills")
    years_of_experience: float = Field(description="Total professional years of experience as a float")
    work_history: List[WorkExperience] = Field(default=[], description="Past work history entries")
    
    # Portfolio, Projects & Extracurriculars
    linkedin: Optional[str] = Field(default="", description="LinkedIn profile URL")
    github: Optional[str] = Field(default="", description="GitHub profile URL or username")
    languages: List[str] = Field(default=[], description="Spoken and written languages")
    projects: List[ProjectInfo] = Field(default=[], description="Academic, personal, or open-source projects")
    leadership_activities: List[str] = Field(default=[], description="Co-curricular activities, leadership roles, or volunteering")
    awards: List[str] = Field(default=[], description="Competitions, scholarships, honors, or awards")

class CandidateUpdate(BaseModel):
    name: str
    email: str = ""
    phone: str = ""
    location: str
    education_level: str
    highest_education: str
    skills: List[str] = Field(default=[])
    work_history: List[WorkExperience] = Field(default=[])
    years_of_experience: float
    linkedin: Optional[str] = ""
    github: Optional[str] = ""
    languages: List[str] = Field(default=[])
    projects: List[ProjectInfo] = Field(default=[])
    leadership_activities: List[str] = Field(default=[])
    awards: List[str] = Field(default=[])

class CandidateEvaluation(BaseModel):
    name: str = Field(description="Full name of candidate")
    match_level: Literal["High", "Medium", "Low"] = Field(description="Overall categorical alignment")
    fit_score: int = Field(description="Suitability rating from 0 to 100")
    education_fit: str = Field(description="Assessment of degree and qualification relevance")
    key_strengths: List[str] = Field(description="Specific matching qualifications, skills, and standout projects/awards")
    gaps_or_risks: List[str] = Field(description="Missing requirements or qualification gaps")
    recommendation: str = Field(description="Actionable decision: e.g. 'Fast-track to interview', 'Reject'")

class SourcingEvaluationResult(BaseModel):
    query_summary: str = Field(description="Concise summary of criteria evaluated")
    evaluations: List[CandidateEvaluation] = Field(description="Evaluations for candidates")

class SearchRequest(BaseModel):
    query: str
    candidate_pool: List[dict]

SYSTEM_HR_INSTRUCTION = """
You are an expert HR Talent Acquisition Specialist and Senior Technical Recruiter.
Rules:
1. Always normalize 'education_level' strictly into: 'High School', 'Diploma', 'Bachelor', 'Master', or 'PhD'.
2. Aggregate work history durations accurately into 'years_of_experience' as a float.
3. Be strictly factual; never hallucinate unstated skills, projects, or achievements.
4. When evaluating candidates against job requirements, strictly check hard criteria (required skills, minimum experience years, education).
"""

# =============================================================
# 4. API ENDPOINTS
# =============================================================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Gemini HR Agent API",
        "firebase_connected": db is not None,
        "gemini_connected": client is not None
    }


@app.post("/api/upload-and-parse-resume")
async def upload_and_parse_resume(file: UploadFile = File(...)):
    """
    1. Saves uploaded resume to Firebase Storage.
    2. Uses Gemini 3.8 Flash to extract structured candidate JSON.
    3. Persists document into Firestore 'candidates' collection with resume_url.
    """
    if not client:
        raise HTTPException(status_code=500, detail="Gemini client is not initialized. Check GEMINI_API_KEY.")
    if not db or not bucket:
        raise HTTPException(status_code=500, detail="Firebase is not initialized. Check Firebase environment variables.")

    candidate_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[-1].lower()
    storage_blob_path = f"resumes/{candidate_id}_{file.filename}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # A. Upload original file to Firebase Storage
        blob = bucket.blob(storage_blob_path)
        blob.upload_from_filename(temp_path, content_type=file.content_type)
        blob.make_public()
        resume_url = blob.public_url

        # B. Parse resume with Gemini Multimodal File API
        uploaded_doc = client.files.upload(file=temp_path)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_HR_INSTRUCTION,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=CandidateProfile,
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[uploaded_doc, "Analyze this resume and extract candidate profile details, projects, awards, and portfolios."],
            config=config
        )

        candidate_data = json.loads(response.text)

        # C. Attach storage and metadata tracking
        candidate_data["candidate_id"] = candidate_id
        candidate_data["resume_url"] = resume_url
        candidate_data["resume_storage_path"] = storage_blob_path
        candidate_data["original_filename"] = file.filename
        candidate_data["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        candidate_data["status"] = "active"

        # D. Save directly to Firestore 'candidates' collection
        db.collection("candidates").document(candidate_id).set(candidate_data)

        # Clean up temporary Gemini uploaded file reference
        try:
            client.files.delete(name=uploaded_doc.name)
        except Exception:
            pass

        return {
            "status": "success",
            "message": "Resume uploaded, parsed, and saved to Firestore successfully.",
            "candidate_id": candidate_id,
            "data": candidate_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.put("/api/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, payload: CandidateUpdate):
    """Updates recruiter-corrected profile fields in the candidates collection."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase is not initialized.")

    candidate_ref = db.collection("candidates").document(candidate_id)
    if not candidate_ref.get().exists:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    candidate_ref.update({
        **payload.model_dump(),
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    })

    return {
        "status": "success",
        "message": "Candidate profile updated successfully.",
        "candidate_id": candidate_id,
        "data": payload.model_dump(),
    }


@app.post("/api/find-candidates")
def find_candidates_endpoint(payload: SearchRequest):
    """
    Evaluates candidate profiles against an HR query and outputs structured JSON fit evaluations.
    Takes into account skills, work history, relevant projects, github/linkedin, and awards.
    """
    if not client:
        raise HTTPException(status_code=500, detail="Gemini client is not initialized.")

    if not payload.candidate_pool:
        return {"query_summary": "Empty candidate pool", "evaluations": []}

    try:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_HR_INSTRUCTION,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=SourcingEvaluationResult,
        )

        prompt = f"""
Job Requirement / Sourcing Query:
"{payload.query}"

Candidate Pool:
{json.dumps(payload.candidate_pool, indent=2)}

Task:
1. Extract explicit must-have requirements (skills, minimum experience, education, languages).
2. Objectively score and evaluate EVERY candidate in the candidate pool.
3. Compute a realistic fit_score (0-100) and assign High/Medium/Low match_level.
4. When evaluating strengths and calculating fit_score, consider:
   - Relevance and complexity of candidate projects.
   - GitHub or portfolio evidence demonstrating the required skills.
   - Relevant honors, awards, leadership activities, or languages.
5. Highlight concrete strengths and specific gaps.
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=config
        )

        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Candidate matching error: {str(e)}")
import os
import json
import uuid
import shutil
import tempfile
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, File, Header, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials, firestore, storage

# 1. APPLICATION & CORS SETUP
app = FastAPI(title="Gemini HR Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. FIREBASE & GEMINI CLIENT INITIALIZATION
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

# 3. PYDANTIC SCHEMAS
class WorkExperience(BaseModel):
    role: str = Field(description="Job title or designation")
    company: str = Field(description="Company or organization name")
    duration: str = Field(description="Dates or duration of employment")

class ProjectInfo(BaseModel):
    title: str = Field(description="Project title or name")
    description: str = Field(description="Brief overview, technologies used, or outcome of the project")

class CandidateProfile(BaseModel):
    is_valid_resume: bool = Field(description="Set to True if the document is a resume/CV. " \
    "Set to False if it is a random document, menu, image, or unrelated text.")
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

class UserAccountRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str

# HELPER FUNCTIONS
def get_hr_instruction():
    """Generates prompt with current date so Gemini calculates 'Present' experience correctly."""
    current_date = datetime.utcnow().strftime("%B %Y")
    return f"""
You are an expert HR Talent Acquisition Specialist and Senior Technical Recruiter.
Today's date is {current_date}. 
Rules:
1. FIRST, determine if the uploaded document is actually a resume, CV, or professional profile. If it is NOT (e.g., it's a receipt, manual, or random image), set `is_valid_resume` to False and leave other fields blank.
2. Always normalize 'education_level' strictly into: 'High School', 'Diploma', 'Bachelor', 'Master', or 'PhD'.
3. Aggregate work history durations accurately into 'years_of_experience' as a float. Use today's date for 'Present' roles.
4. Be strictly factual; never hallucinate unstated skills, projects, or achievements.
5. When evaluating candidates against job requirements, strictly check hard criteria (required skills, minimum experience years, education).
"""

def clean_json_response(raw_text: str) -> dict:
    """Strips Markdown backticks if Gemini accidentally includes them, preventing JSONDecodeError."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
        
    return json.loads(text.strip())


# 4. API ENDPOINTS
def require_super_admin(authorization: Optional[str]) -> dict:
    if not db or not firebase_admin._apps:
        raise HTTPException(status_code=500, detail="Firebase is not initialized.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication is required.")

    try:
        token = firebase_auth.verify_id_token(authorization[7:])
    except Exception as error:
        raise HTTPException(status_code=401, detail="Invalid authentication token.") from error

    if token.get("email") == "superadmin@centific.com":
        return token

    profile = db.collection("users").document(token["uid"]).get()
    profile_data = profile.to_dict() if profile.exists else {}
    if profile_data.get("role") == "Super Admin" or profile_data.get("roles") == "Super Admin":
        return token

    raise HTTPException(status_code=403, detail="Super Admin access is required.")


@app.post("/api/users")
def create_user_account(
    payload: UserAccountRequest,
    authorization: Optional[str] = Header(default=None),
):
    require_super_admin(authorization)
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    try:
        account = firebase_auth.create_user(
            email=payload.email.strip().lower(),
            password=payload.password,
            display_name=payload.name.strip(),
        )
        db.collection("users").document(account.uid).set({
            "name": payload.name.strip(),
            "email": payload.email.strip().lower(),
            "roles": payload.role,
        })
        return {"id": account.uid}
    except Exception as error:
        raise HTTPException(status_code=400, detail="Unable to create the user account.") from error


# Converted to path parameter so frontend fetch() calls don't crash when passing a body.
@app.delete("/api/users/{user_id}")
def delete_user_account(
    user_id: str,
    email: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    require_super_admin(authorization)
    uid = user_id

    try:
        firebase_auth.get_user(uid)
    except firebase_auth.UserNotFoundError:
        if not email:
            raise HTTPException(status_code=404, detail="User account was not found.")
        try:
            uid = firebase_auth.get_user_by_email(email).uid
        except firebase_auth.UserNotFoundError as error:
            raise HTTPException(status_code=404, detail="User account was not found.") from error

    firebase_auth.delete_user(uid)
    db.collection("users").document(user_id).delete()
    if uid != user_id:
        db.collection("users").document(uid).delete()
    return {"id": uid}


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
    if not client:
        raise HTTPException(status_code=500, detail="Gemini client is not initialized. Check GEMINI_API_KEY.")
    if not db or not bucket:
        raise HTTPException(status_code=500, detail="Firebase is not initialized.")

    # We temporarily use a random UUID for the storage file name to avoid file overwriting conflicts
    temp_storage_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[-1].lower()
    storage_blob_path = f"resumes/{temp_storage_id}_{file.filename}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # A. Upload original file to Firebase Storage
        blob = bucket.blob(storage_blob_path)
        blob.upload_from_filename(temp_path, content_type=file.content_type)
        blob.make_public()
        resume_url = blob.public_url

        # B. Parse resume with Gemini
        uploaded_doc = client.files.upload(file=temp_path)

        config = types.GenerateContentConfig(
            system_instruction=get_hr_instruction(),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=CandidateProfile,
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[uploaded_doc, "Analyze this resume and extract candidate profile details, projects, awards, and portfolios."],
            config=config
        )

        candidate_data = clean_json_response(response.text)
        
        # Check if it's a valid resume
        if not candidate_data.get("is_valid_resume", True):
            # Delete the useless file from Firebase Storage
            blob.delete()
            # Delete from Gemini
            try:
                client.files.delete(name=uploaded_doc.name)
            except Exception:
                pass
            # Throw error to the frontend
            raise HTTPException(status_code=400, detail="The uploaded file does not appear to be a valid resume or CV. Please upload a correct document.")
    
        # Attach new file metadata
        candidate_data["resume_url"] = resume_url
        candidate_data["resume_storage_path"] = storage_blob_path
        candidate_data["original_filename"] = file.filename
        
        # C. DEDUPLICATION LOGIC: Check for ACTIVE candidates by Email
        email = candidate_data.get("email", "").strip()
        existing_active_doc = None
        
        if email:
            # Fetch all past and present profiles with this exact email
            matches = db.collection("candidates").where("email", "==", email).get()
            
            for doc_snapshot in matches:
                doc_status = doc_snapshot.to_dict().get("status", "")
                
                # If they already have an ongoing active/pending profile, we flag it to be updated.
                # If the status is "Hired" or "Reject", we ignore it so a fresh profile gets created.
                if doc_status not in ["Hired", "Reject"]:
                    existing_active_doc = doc_snapshot
                    break

        if existing_active_doc:
            # UPDATE EXISTING ONGOING CANDIDATE
            candidate_id = existing_active_doc.id
            candidate_data["candidate_id"] = candidate_id
            candidate_data["updated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            # Merge prevents overwriting existing internal notes
            db.collection("candidates").document(candidate_id).set(candidate_data, merge=True)
            message = "Candidate already has an active profile. Updated with new resume data."
        else:
            # CREATE NEW CANDIDATE (First time applying, OR previously Hired/Rejected)
            candidate_id = temp_storage_id # Reuse the generated ID
            candidate_data["candidate_id"] = candidate_id
            candidate_data["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            candidate_data["status"] = "active"
            
            db.collection("candidates").document(candidate_id).set(candidate_data)
            message = "New application profile created successfully."

        # Clean up temporary Gemini uploaded file reference
        try:
            client.files.delete(name=uploaded_doc.name)
        except Exception:
            pass

        return {
            "status": "success",
            "message": message,
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
    if not client:
        raise HTTPException(status_code=500, detail="Gemini client is not initialized.")

    active_candidate_pool = [
        candidate for candidate in payload.candidate_pool
        if isinstance(candidate.get("status"), str)
        and candidate["status"].strip().lower() == "active"
    ]

    if not active_candidate_pool:
        return {"query_summary": "Empty candidate pool", "evaluations": []}

    try:
        config = types.GenerateContentConfig(
            system_instruction=get_hr_instruction(), # FIX 2: Dynamic date injected
            temperature=0.0, 
            response_mime_type="application/json",
            response_schema=SourcingEvaluationResult,
        )

        prompt = f"""
Job Requirement / Sourcing Query:
"{payload.query}"

Candidate Pool:
{json.dumps(active_candidate_pool, indent=2)}

Task:
1. Identify the MANDATORY (non-negotiable) hard requirements from the query (e.g., specific programming languages, technical skills, degree level).
2. HARD FILTER / ELIMINATION RULE:
   - If a candidate completely lacks the mandatory technical skill (e.g., the query requires C++, but the candidate has only Business/Finance skills and zero C++ experience), DO NOT INCLUDE THEM in the `evaluations` list.
   - Do NOT lower the hiring standard just because the candidate pool is small.
   - It is completely acceptable to return an empty `evaluations: []` list if no candidates meet the minimum baseline.
3. For candidates who DO possess the baseline skills:
   - Compute a realistic fit_score (0-100) and assign High, Medium, or Low match_level.
   - Consider relevant projects, work history, awards, and portfolios.
   - Highlight concrete strengths and gaps.
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=config
        )

        # Safe JSON parsing
        result = clean_json_response(response.text)

        result["evaluations"] = [
            cand for cand in result.get("evaluations", [])
            if cand.get("match_level") in ["High", "Medium"] and cand.get("fit_score", 0) >= 40
        ]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Candidate matching error: {str(e)}")
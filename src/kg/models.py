from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Researcher:
    id: str
    name: str
    university: str
    email: Optional[str] = None
    department: Optional[str] = None
    google_scholar_id: Optional[str] = None
    github_username: Optional[str] = None
    linkedin_url: Optional[str] = None
    created_at: Optional[str] = datetime.now().isoformat()
    updated_at: Optional[str] = datetime.now().isoformat()

@dataclass
class Paper:
    id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
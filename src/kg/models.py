from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum, auto

@dataclass
class Researcher:
    id: str
    name: str
    university: str
    email: Optional[str] = None
    google_scholar_id: Optional[str] = None
    github_username: Optional[str] = None
    linkedin_url: Optional[str] = None
    personal_website: Optional[str] = None
    research_areas: List[str] = Dict[research_hierarchy]
    created_at: Optional[str] = datetime.now().isoformat()
    updated_at: Optional[str] = datetime.now().isoformat()
    confidence_score: float = 0.0

@dataclass
class Paper:
    id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: Optional[str] = None
    doi: Optional[str] = None

@dataclass
class Repository:
    id: str
    name: str
    owner: str
    description: Optional[str] = None
    stars: int = 0
    forks: int = 0
    topics: Dict[research_hierarchy]
    created_at: Optional[str] = datetime.now().isoformat()
    updated_at: Optional[str] = datetime.now().isoformat()

research_hierarchy = {
    "Artificial Intelligence": {
        "Machine Learning": [
            "Supervised Learning",
            "Unsupervised Learning",
            "Reinforcement Learning",
            "Self-Supervised Learning",
            "Few-shot Learning",
            "Zero-shot Learning"
        ],
        "Deep Learning": [
            "Convolutional Neural Networks (CNNs)",
            "Recurrent Neural Networks (RNNs)",
            "Transformers",
            "Generative Adversarial Networks (GANs)",
            "Variational Autoencoders (VAEs)"
        ],
        "Natural Language Processing": [
            "Large Language Models (LLMs)"
        ],
        "Computer Vision": [
            "Object Detection",
            "Image Segmentation",
            "Image Generation",
            "Video Understanding"
        ],
        "Explainable AI": [
            "Causal Explainability"
        ],
        "Cybersecurity": [
            "Data Poisoning",
            "Adversarial Machine Learning",
            "Privacy-Preserving Machine Learning",
            "Program Analysis"
        ],
    },
    "Robotics": {
        "Robot Perception": [
            "Sensor Fusion",
            "SLAM (Simultaneous Localisation and Mapping)",
            "Visual Perception"
        ],
        "Robot Control": [
            "Motion Planning",
            "Trajectory Optimisation",
            "Model Predictive Control (MPC)"
        ],
        "Multi-Robot Systems": [
            "Swarm Robotics",
            "Coordination and Cooperation"
        ],
        "Robot Learning": [
            "Imitation Learning",
            "Reinforcement Learning in Robotics",
            "Sim2Real Transfer"
        ],
        "Soft Robotics": [
            "Bio-Inspired Design",
            "Soft Actuators"
        ],
    },
    "Multi-DomainAreas": {
        "": [ 
            "Vision + Language",
            "Speech + Text",
            "Embodied AI",
            "Neurosymbolic AI",
            "Digital Twins / Simulation",
            "Chip Design",
            "Edge AI Devices",
        ]
    }
}

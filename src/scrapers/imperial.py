from kg.db import Neo4jConnection
from kg.manager import KnowledgeGraphManager
from kg.models import Researcher, Paper, Repository, research_hierarchy
import uuid

def create_imperial_researcher(name: str, email: str, department: str, position: str, research_areas: Dict[research_hierarchy], **kwargs) -> Researcher:
    """Create an ImperialResearcher instance with a unique ID."""
    return Researcher(
        id=str(uuid.uuid4()),
        name=name,
        email=email,
        university="Imperial College London",
        research_areas=research_areas,
        **kwargs
    )

def main():
    researchers = [
        create_imperial_researcher(
            name="John Smith",
            email="john.smith@imperial.ac.uk",
            google_scholar_id="abc123",
            github_username="jsmith",
            personal_website="https://www.imperial.ac.uk/people/john.smith",
            research_areas=research_hierarchy["Artificial Intelligence"]["Deep Learning"]
        ),
        create_imperial_researcher(
            name="Jane Doe",
            email="jane.doe@imperial.ac.uk",
            google_scholar_id="def456",
            github_username="jdoe",
            personal_website="https://www.imperial.ac.uk/people/jane.doe",
            research_areas=research_hierarchy["Natural Language Processing"]["Large Language Models (LLMs)"]
        )
    ]

    with Neo4jConnection() as conn:
        kg = KnowledgeGraphManager(conn)
        
        # Create researchers in the graph
        for researcher in researchers:
            try:
                result = kg.create_researcher(researcher)
                print(f"Created researcher: {researcher.name}")
            except Exception as e:
                print(f"Error creating researcher {researcher.name}: {str(e)}")

if __name__ == "__main__":
    main()
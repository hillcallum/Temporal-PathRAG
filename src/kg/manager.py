from .db import Neo4jConnection
from .models import Researcher, Paper

class KnowledgeGraphManager:
    def __init__(self, connection: Neo4jConnection):
        self.conn = connection

    def create_researcher(self, researcher: Researcher):
        query = """
        CREATE (r:Researcher {
            id: $id,
            name: $name,
            university: $university,
            email: $email,
            department: $department,
            google_scholar_id: $google_scholar_id,
            github_username: $github_username,
            linkedin_url: $linkedin_url,
            created_at: $created_at,
            updated_at: $updated_at
        })
        RETURN r
        """
        params = {
            'id': researcher.id,
            'name': researcher.name,
            'university': researcher.university,
            'email': researcher.email,
            'department': researcher.department,
            'google_scholar_id': researcher.google_scholar_id,
            'github_username': researcher.github_username,
            'linkedin_url': researcher.linkedin_url,
            'created_at': researcher.created_at,
            'updated_at': researcher.updated_at
        }
        return self.conn.execute_query(query, params)
    
    def get_researcher(self, researcher_id: str):
        query = "MATCH (r:Researcher {id: $id}) RETURN r"
        result = self.conn.execute_query(query, {'id': researcher_id})
        return result[0] if result else None
    
    def update_researcher(self, researcher_id: str, updates: dict):
        set_clause = ", ".join([f"r.{k} = ${k}" for k in updates])
        query = f"""
        MATCH (r:Researcher {{id: $researcher_id}})
        SET {set_clause}, r.updated_at = datetime()
        RETURN r
        """
        params = {'researcher_id': researcher_id, **updates}
        return self.conn.execute_query(query, params)
    
    def create_paper(self, paper: Paper):
        query = """
        CREATE (p:Paper {
            id: $id,
            title: $title,
            authors: $authors,
            year: $year,
            venue: $venue,
            abstract: $abstract,
            doi: $doi
        })
        RETURN p
        """
        params = {
            'id': paper.id,
            'title': paper.title,
            'authors': paper.authors,
            'year': paper.year,
            'venue': paper.venue,
            'abstract': paper.abstract,
            'doi': paper.doi
        }
        return self.conn.execute_query(query, params)
    
    def get_paper(self, paper_id: str):
        query = "MATCH (p:Paper {id: $id}) RETURN p"
        result = self.conn.execute_query(query, {'id': paper_id})
        return result[0] if result else None
    
    def update_paper(self, paper_id: str, updates: dict):
        set_clause = ", ".join([f"p.{k} = ${k}" for k in updates])
        query = f"""
        MATCH (p:Paper {{id: $paper_id}})
        SET {set_clause}, p.updated_at = datetime()
        RETURN p
        """
        params = {'paper_id': paper_id, **updates}
        return self.conn.execute_query(query, params)
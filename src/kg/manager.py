from .db import Neo4jConnection
from .models import Researcher, Paper, Repository, research_hierarchy

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
            google_scholar_id: $google_scholar_id,
            github_username: $github_username,
            linkedin_url: $linkedin_url,
            personal_website: $personal_website,
            research_areas: $research_areas,
            created_at: $created_at,
            updated_at: $updated_at,
            confidence_score: $confidence_score
        })
        RETURN r
        """
        params = {
            'id': researcher.id,
            'name': researcher.name,
            'university': researcher.university,
            'email': researcher.email,
            'google_scholar_id': researcher.google_scholar_id,
            'github_username': researcher.github_username,
            'linkedin_url': researcher.linkedin_url,
            'personal_website': researcher.personal_website,
            'research_areas': researcher.research_areas,
            'created_at': researcher.created_at,
            'updated_at': researcher.updated_at,
            'confidence_score': researcher.confidence_score
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

    def create_repository(self, repo: Repository):
        query = """
        CREATE (r:Repository {
            id: $id,
            name: $name,
            owner: $owner,
            description: $description,
            stars: $stars,
            forks: $forks,
            topics: $topics,
            created_at: $created_at,
            updated_at: $updated_at
        })
        RETURN r
        """
        params = {
            'id': repo.id,
            'name': repo.name,
            'owner': repo.owner,
            'description': repo.description,
            'stars': repo.stars,
            'forks': repo.forks,
            'topics': repo.topics,
            'created_at': repo.created_at,
            'updated_at': repo.updated_at
        }
        return self.conn.execute_query(query, params)
    
    def link_researcher_to_paper(self, researcher_id: str, paper_id: str, is_author: bool = True):
        relationship = "AUTHORED" if is_author else "CITED"
        query = f"""
        MATCH (r:ImperialResearcher {{id: $researcher_id}})
        MATCH (p:Paper {{id: $paper_id}})
        CREATE (r)-[rel:{relationship}]->(p)
        RETURN r, rel, p
        """
        return self.conn.execute_query(query, {
            'researcher_id': researcher_id,
            'paper_id': paper_id
        })
    
    def link_researcher_to_repository(self, researcher_id: str, repo_id: str):
        query = """
        MATCH (r:ImperialResearcher {id: $researcher_id})
        MATCH (repo:Repository {id: $repo_id})
        CREATE (r)-[rel:CONTRIBUTES_TO]->(repo)
        RETURN r, rel, repo
        """
        return self.conn.execute_query(query, {
            'researcher_id': researcher_id,
            'repo_id': repo_id
        })
    
    def get_researcher_by_area(self, research_area: ResearchArea):
        query = """
        MATCH (r:ImperialResearcher)
        WHERE $research_area IN r.research_areas
        RETURN r
        """
        return self.conn.execute_query(query, {'research_area': research_area.value})
    
    def update_researcher_confidence(self, researcher_id: str, new_score: float):
        query = """
        MATCH (r:ImperialResearcher {id: $researcher_id})
        SET r.confidence_score = $new_score,
            r.updated_at = datetime(),
            r.last_verified = datetime()
        RETURN r
        """
        return self.conn.execute_query(query, {
            'researcher_id': researcher_id,
            'new_score': new_score
        })
    
    def get_researcher_network(self, researcher_id: str, depth: int = 2):
        query = """
        MATCH path = (r:ImperialResearcher {id: $researcher_id})-[*1..$depth]-(connected)
        RETURN path
        """
        return self.conn.execute_query(query, {
            'researcher_id': researcher_id,
            'depth': depth
        })
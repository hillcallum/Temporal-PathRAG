from kg.db import Neo4jConnection
from kg.manager import KnowledgeGraphManager
from scrapers.imperial import ImperialScraper
from scrapers.scholar import ScholarScraper
# from scrapers.github import GithubScraper

def main():
    with Neo4jConnection() as conn:
        kg = KnowledgeGraphManager(conn)
        imperial = ImperialScraper()
        scholar = ScholarScraper()

        faculty = imperial.get_faculty_list()
        for person in faculty:
            scholar_data = scholar.get_researcher_profile(person['name'])
            # Merge data, create Researcher, insert into KG
            # TODO: Add deduplication, error handling, etc.

if __name__ == "__main__":
    main()
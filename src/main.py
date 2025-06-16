from kg.db import Neo4jConnection
from kg.manager import KnowledgeGraphManager

def main():
    with Neo4jConnection() as conn:
        kg = KnowledgeGraphManager(conn)
        # Example usage
        print("Knowledge Graph Manager initialized")

if __name__ == "__main__":
    main()
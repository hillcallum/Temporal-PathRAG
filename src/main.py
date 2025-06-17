from kg.db import Neo4jConnection
from kg.manager import KnowledgeGraphManager

def main():
    with Neo4jConnection() as conn:
        kg = KnowledgeGraphManager(conn)
        print("Knowledge Graph Manager initialised")

if __name__ == "__main__":
    main()
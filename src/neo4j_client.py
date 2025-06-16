from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def test_connection(self):
        with self.driver.session() as session:
            result = session.run("RETURN 'Connected to Neo4j successfully!'")
            for record in result:
                print(record)

    def create_university(self, name):
        with self.driver.session() as session:
            session.run(
                "CREATE (u:University {name: $name})",
                name=name
            )


if __name__ == "__main__":
    client = Neo4jClient()
    client.test_connection()
    client.create_university("Imperial College London")
    client.close()
        
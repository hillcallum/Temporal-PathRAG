from src.neo4j_client import driver

def create_university(name):
    query = """
    CREATE (u:University {name: $name})
    RETURN u
    """
    with driver.session() as session:
        result = session.run(query, {"name": name})
        print(result.single().get("u"))
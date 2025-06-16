from kg.db import Neo4jConnection

def test_connection():
    with Neo4jConnection() as conn:
        result = conn.execute_query("RETURN 1 as test")
        assert result[0]['test'] == 1
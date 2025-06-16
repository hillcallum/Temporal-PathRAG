from kg.db import Neo4jConnection

CONSTRAINTS = [
    "CREATE CONSTRAINT researcher_id IF NOT EXISTS FOR (r:Researcher) REQUIRE r.id IS UNIQUE;",
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE;",
    "CREATE CONSTRAINT repo_id IF NOT EXISTS FOR (rep:Repository) REQUIRE rep.id IS UNIQUE;",
    "CREATE INDEX researcher_name IF NOT EXISTS FOR (r:Researcher) ON (r.name);",
    "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title);",
    "CREATE INDEX researcher_department IF NOT EXISTS FOR (r:Researcher) ON (r.department);"
]

def setup_constraints():
    with Neo4jConnection() as conn:
        for query in CONSTRAINTS:
            conn.execute_query(query)
        print("Constraints and indexes created.")

if __name__ == "__main__":
    setup_constraints()
"""
Optional Neo4j persistence/sync layer.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from config.settings import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


class Neo4jStore:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
    ):
        self.database = database
        self._driver = None
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            logger.info(f"Neo4jStore connected | uri={uri} db={database}")
        except Exception as exc:
            logger.warning(f"Neo4j unavailable, continuing with in-memory graph: {exc}")

    @property
    def enabled(self) -> bool:
        return self._driver is not None

    def close(self):
        if self._driver:
            self._driver.close()

    def upsert_person(self, person_id: str, last_seen: float):
        if not self._driver:
            return
        query = """
        MERGE (p:Person {person_id: $person_id})
        ON CREATE SET p.first_seen = $last_seen
        SET p.last_seen = $last_seen
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, person_id=person_id, last_seen=float(last_seen))

    def upsert_relationship(self, pid_a: str, pid_b: str, confidence: float, relationship: str, last_incident: str):
        if not self._driver:
            return
        query = """
        MERGE (a:Person {person_id: $pid_a})
        MERGE (b:Person {person_id: $pid_b})
        MERGE (a)-[r:RELATES_TO]-(b)
        SET r.confidence = $confidence,
            r.relationship = $relationship,
            r.last_incident = $last_incident,
            r.last_seen = timestamp() / 1000.0
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                pid_a=pid_a,
                pid_b=pid_b,
                confidence=float(confidence),
                relationship=relationship,
                last_incident=last_incident,
            )

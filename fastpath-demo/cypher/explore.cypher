// FastPath finance demo: queries for Neo4j Explore / Query (Cypher 25)

// 1. One asset's event path through the tariff week, with the Moments it joined
CYPHER 25
MATCH (a:Asset {ticker: 'NVDA'})-[:FIRST_EVENT]->(:MarketEvent)-[:NEXT]->*(e:MarketEvent)
WHERE '2025-04-01' <= e.date <= '2025-04-11'
OPTIONAL MATCH (e)-[r:IN_MOMENT]->(m:Moment)
RETURN e, r, m;

// 2. The biggest Moments of the tariff shock and who was in them
CYPHER 25
MATCH (m:Moment) WHERE '2025-04-02' <= m.date <= '2025-04-10'
ORDER BY m.size DESC LIMIT 3
MATCH (e:MarketEvent)-[r:IN_MOMENT]->(m)
RETURN m, r, e;

// 3. Why are two assets neighbours as of a date? Shared Moments in the 180-day look-back
CYPHER 25
WITH 'NVDA' AS a, 'AVGO' AS b, '2025-03-31' AS asOf
MATCH (s:Snapshot {ticker: a, date: asOf})
MATCH (ea:MarketEvent {ticker: a})-[:IN_MOMENT]->(m:Moment)<-[:IN_MOMENT]-(eb:MarketEvent {ticker: b})
WHERE s.observationDay - 180 < m.day <= s.observationDay
RETURN m.date AS date, m.direction AS together, m.size AS assetsInMoment
ORDER BY date DESC;

// 4. Moment overlap ranking for one asset (the graph-only baseline, in Cypher)
CYPHER 25
WITH 'NVDA' AS focus, '2025-03-31' AS asOf
MATCH (s:Snapshot {ticker: focus, date: asOf})
MATCH (:MarketEvent {ticker: focus})-[:IN_MOMENT]->(m:Moment)<-[:IN_MOMENT]-(o:MarketEvent)
WHERE s.observationDay - 180 < m.day <= s.observationDay
RETURN o.ticker AS other, count(m) AS sharedMoments
ORDER BY sharedMoments DESC LIMIT 15;

// 5. After the notebook wrote embeddings: nearest neighbours as of a date
CYPHER 25
MATCH (s:Snapshot {ticker: 'JPM', date: '2025-03-31'})
CALL db.index.vector.queryNodes('snapshot_fastpath', 1000, s.fastPathEmbedding) YIELD node, score
WHERE node.date = s.date AND node <> s
RETURN node.ticker AS ticker, round(score, 3) AS similarity
LIMIT 10;

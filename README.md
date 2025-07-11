# Neo4j Quantitative Analysis Demo

This repository contains a Neo4j-based demo for quantitative analysis of financial assets. It demonstrates how to model temporal price data, compute correlations between assets using graph data science (GDS), and persist these correlations over time. The focus is on handling time-series data for assets, calculating log returns, and using KNN with Pearson similarity to identify correlated assets.

## Context

In quantitative finance, analyzing correlations between assets is crucial for portfolio management, risk assessment, and trading strategies like pairs trading. This demo uses Neo4j to store asset price data as a graph, where:
- Assets have daily price nodes (`AssetDay`) chained by time.
- Correlations are computed dynamically using sliding windows of historical log returns.
- Neo4j's Graph Data Science (GDS) library is leveraged for efficient similarity calculations (e.g., Pearson correlation via KNN).

The data is based on real historical stock prices (fetched via yfinance in a separate Python script, not included here), covering ~100 assets across categories like Tech, Finance, Energy, Healthcare, and Consumer. The model supports "as-of-date" snapshots for backtesting or real-time analysis.

Key features:
- Temporal chaining of price data.
- Precomputed correlations persisted as relationships.
- Support for variable window sizes (e.g., 30 days).
- Visualization of correlation graphs.

This demo was built iteratively, addressing challenges like handling irregular time series, efficient computation, and scalability on Neo4j AuraDS (with specs like 6 CPUs, 32GB RAM).

## Problem Statement

The core challenge is building a graph data model for quantitative analysis that:
- Captures temporal evolution of asset prices (e.g., OHLCV and adjusted close).
- Computes correlations between assets' time series in a sliding window (e.g., last 30 days' log returns).
- Supports efficient "as-of-date" queries for snapshots (e.g., what were the correlations on a specific date?).
- Scales for batch processing across all historical dates without excessive query times.
- Handles edge cases like insufficient historical data (e.g., early dates with <30 days prior).

Initial model lacked temporal handling, so we introduced chained `AssetDay` nodes and used GDS for on-the-fly or precomputed correlations. Computations must be efficient (e.g., <100ms per date, <10s for batch), using log returns to stationarize data for Pearson correlation.

## Data Model

- **AssetDay Nodes**: Represent daily prices for an asset.
  - Properties: `asset_id`, `category`, `ticker`, `name`, `date` (YYYY-MM-DD), `open`, `high`, `low`, `close`, `volume`, `adj_close`, `asset_day_id` (unique key like `asset_id_date`).
  - Chained via `[:NEXT]` relationships for time traversal.

- **CORRELATES_WITH Relationships**: Persisted correlations between `AssetDay` nodes.
  - Properties: `pearson_correlation_score` (similarity score from KNN).

Data import (not in queries): Use CSV with columns like `asset_id, category, ticker, name, date, open, high, low, close, volume, adj_close`. Load via Cypher `LOAD CSV` and create nodes/links.

## Queries and Explanations

The queries are structured as a tree in the provided CSV (`neo4j_query_saved_cypher_2025-7-10 (2).csv`). Below is an explanation of each, including purpose and key Cypher elements. All queries assume APOC and GDS libraries are installed.

### 1. **set params**
   - **Query**:
     ```cypher
     :params {
       date: datetime("2025-06-09T00:00:00Z"),
       window_days: 30
     }
     ```
   - **Explanation**: Sets global parameters for date (as-of-date for snapshots) and window_days (e.g., 30 for correlation window). Used in subsequent queries for flexibility.

### 2. **chain AssetDays**
   - **Query**:
     ```cypher
     CYPHER 25
     MATCH (ad:AssetDay)
     ORDER BY ad.date
     WITH ad.asset_id AS id, collect(ad) AS ads
     CALL apoc.nodes.link(ads, 'NEXT')
     ```
   - **Explanation**: Chains `AssetDay` nodes per asset using `[:NEXT]` relationships, ordered by date. This creates temporal linked lists for efficient traversal (e.g., finding previous 30 days). APOC is used for bulk linking.

### 3. **project into memory for date**
   - **Query**:
     ```cypher
     CYPHER 25
     MATCH (ad:AssetDay {date: $date})
     CALL (ad) {
     MATCH ((prev_days:AssetDay)-[:NEXT]->(:AssetDay)){30}(ad)
     WITH ad, apoc.coll.zip(prev_days, prev_days[1..]+[ad]) AS t_tplus1
     WITH [x IN t_tplus1 | log(x[1].adj_close / x[0].adj_close)] AS log_returns
     RETURN log_returns
     }
     RETURN ad, log_returns
     NEXT
     RETURN gds.graph.project(
       'myGraph',
       ad,
       null,
       {
         sourceNodeProperties: {log_returns: log_returns},
         targetNodeProperties: {}
       }
     )
     ```
   - **Explanation**: For a given `$date`, computes log returns vectors (array of ln(adj_close_t / adj_close_{t-1})) over the last 30 days per asset. Projects these as isolated nodes in an in-memory GDS graph ('myGraph') with `log_returns` as a vector property. Prepares for KNN similarity computation.

### 4. **view slice with correlation**
   - **Query**:
     ```cypher
     CYPHER 25
     // Deleting in memory graph if exists
     OPTIONAL CALL gds.graph.drop('myGraph', false)
     YIELD graphName
     RETURN 0 AS ok_anyway
     NEXT
     // Computing log returns float array of last $window_days days
     MATCH (ad:AssetDay {date: $date})
     CALL (ad) {
     MATCH ((prev_days:AssetDay)-[:NEXT]->(:AssetDay)){365}(ad)
     WITH ad, apoc.coll.zip(prev_days[365-$window_days..], prev_days[365-$window_days+1..]+[ad]) AS t_tplus1
     WITH [x IN t_tplus1 | log(x[1].adj_close / x[0].adj_close)] AS log_returns
     RETURN log_returns
     }
     RETURN ad, log_returns
     NEXT
     // projecting in-memory set of isolated nodes with vector
     RETURN gds.graph.project(
       'myGraph',
       ad,
       null,
       {
         sourceNodeProperties: {log_returns: log_returns},
         targetNodeProperties: {}
       }
     ) AS g
     NEXT
     // returning virtual knn graph of Pearson correlations as of $date
     CALL gds.knn.stream('myGraph', {
         topK: 1,
         nodeProperties: [{log_returns:"PEARSON"}],
         concurrency: 6,
         sampleRate: 1.0,
         deltaThreshold: 0.0
     })
     YIELD node1, node2, similarity
     WITH gds.util.asNode(node1) AS ad1, gds.util.asNode(node2) AS ad2, similarity
     RETURN ad1, ad2, apoc.create.vRelationship(ad1,'CORRELATES_WITH', {score: similarity}, ad2)
     ```
   - **Explanation**: Drops existing graph, computes log returns (flexible window via slicing a 365-day path), projects to memory, and runs KNN stream to return virtual `CORRELATES_WITH` relationships (topK=1 for demo). Visualizes correlations as a graph. Runs in ~68-100ms.

### 5. **persist all correlations 30 days window**
   - **Query**:
     ```cypher
     CYPHER 25
     MATCH (ad:AssetDay)
     RETURN DISTINCT ad.date AS date ORDER BY date ASC
     NEXT
     CALL (date) {
       // Deleting in memory graph if exists
       WITH date
       OPTIONAL CALL gds.graph.drop('graph_' + toString(date), false)
       YIELD graphName
       RETURN 0 AS ok_anyway, date
       
       NEXT
       
       // Computing log returns float array of last 30 days
       MATCH (ad:AssetDay {date: date})
       CALL (ad) {
       MATCH ((prev_days:AssetDay)-[:NEXT]->(:AssetDay)){30}(ad)
       WITH ad, apoc.coll.zip(prev_days, prev_days[1..]+[ad]) AS t_tplus1
       WITH [x IN t_tplus1 | log(x[1].adj_close / x[0].adj_close)] AS log_returns
       RETURN log_returns
       }
       RETURN ad, log_returns, date
       
       NEXT
       
       // projecting in-memory set of isolated nodes with vector
       RETURN gds.graph.project(
         'graph_' + toString(date),
         ad,
         null,
         {
           sourceNodeProperties: {log_returns: log_returns},
           targetNodeProperties: {}
         }
       ) AS g, date
       
       NEXT
       
       // running virtual knn graph of Pearson correlations as of $date
       CALL gds.knn.write('graph_' + toString(date), {
           writeRelationshipType: 'CORRELATES_WITH',
           writeProperty: 'pearson_correlation_score',
           topK: 2,
           nodeProperties: [{log_returns:"PEARSON"}],
           concurrency: 1,
           sampleRate: 1.0,
           deltaThreshold: 0.0
       })
       YIELD nodesCompared, relationshipsWritten
       RETURN nodesCompared, relationshipsWritten
       NEXT
       OPTIONAL CALL gds.graph.drop('graph_' + toString(date), false)
       YIELD graphName
       RETURN nodesCompared, relationshipsWritten
     } IN CONCURRENT TRANSACTIONS OF 6 ROWS
     RETURN sum(nodesCompared) AS nodesCompared, sum(relationshipsWritten) AS relationshipsWritten
     ```
   - **Explanation**: Batch processes all unique dates: Computes log returns (fixed 30 days), projects per-date graph, runs KNN write to persist `CORRELATES_WITH` rels (topK=2), drops graph. Concurrent (6 rows) for speed (~6-8s total). Aggregates stats at end.

### 6. **clean correlates with**
   - **Query**:
     ```cypher
     MATCH ()-[r:CORRELATES_WITH]->()
     CALL (r) {DELETE r} IN TRANSACTIONS OF 1000 ROWS
     ```
   - **Explanation**: Deletes all persisted `CORRELATES_WITH` relationships in batches (1000 rows) for cleanup/reset.

## Usage

1. **Setup**: Import CSV data into Neo4j, create `AssetDay` nodes, run "chain AssetDays".
2. **Params**: Set `:params` for date/window.
3. **Run Queries**: Use "view slice" for visualization, "persist all" for batch persistence.
4. **Visualize**: In Neo4j Browser, results show correlation graphs (e.g., sector clusters).

## Performance Notes

- Single date: ~100ms.
- Batch all dates: ~6-8s with concurrency.
- Tested on AuraDS (6 CPUs, 32GB RAM).

## Future Extensions

- Add trading rules (e.g., momentum via indicators on prices).
- Integrate portfolios/strategies as nodes.
- Handle high-frequency data with time-trees.

Contributions welcome! For questions, open an issue.

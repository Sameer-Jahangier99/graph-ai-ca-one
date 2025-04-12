from py2neo import Graph, Node, Relationship

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# # Clear existing data if needed
# graph.run("MATCH (n) DETACH DELETE n")

# # Drop existing constraints first, then create them
# graph.run("CALL apoc.schema.assert({}, {})")  # This drops all constraints and indexes

# # Create constraints
# constraints = [
#     "CREATE CONSTRAINT ON (player:Player) ASSERT player.id IS UNIQUE",
#     "CREATE CONSTRAINT ON (club:Club) ASSERT club.id IS UNIQUE",
#     "CREATE CONSTRAINT ON (transfer:Transfer) ASSERT transfer.id IS UNIQUE",
#     "CREATE CONSTRAINT ON (country:Country) ASSERT country.name IS UNIQUE"
# ]

# for constraint in constraints:
#     graph.run(constraint)

# Load players

# graph.run("""
# :auto USING PERIODIC COMMIT 1000
# LOAD CSV WITH HEADERS FROM 'https://s3-eu-west-1.amazonaws.com/football-transfers.neo4j.com/transfers-all.csv' AS row
# MERGE (player:Player {id: row.playerUri}) 
# ON CREATE SET player.name = row.playerName, player.position = row.playerPosition;
# """)


# # Load countries
# graph.run("""
# WITH 'https://s3-eu-west-1.amazonaws.com/football-transfers.neo4j.com/transfers-all.csv' AS url
# LOAD CSV WITH HEADERS FROM url AS row
# WITH row WHERE row.playerNationality <> ''
# WITH DISTINCT row.playerNationality AS nationality
# MERGE (country:Country {name: nationality })
# """)

# # Load clubs and link to countries
# graph.run("""
# WITH 'https://s3-eu-west-1.amazonaws.com/football-transfers.neo4j.com/transfers-all.csv' AS url
# LOAD CSV WITH HEADERS FROM url AS row
# UNWIND [
# {uri: row.sellerClubUri, name: row.sellerClubName, country: row.sellerClubCountry},
# {uri: row.buyerClubUri, name: row.buyerClubName, country: row.buyerClubCountry}
# ] AS club
# WITH club WHERE club.uri <> ''
# WITH DISTINCT club
# MERGE (c:Club {id: club.uri})
# ON CREATE SET c.name = club.name
# MERGE (country:Country {name: club.country })
# MERGE (c)-[:PART_OF]->(country)
# """)

# # Load transfers
# graph.run("""
# USING PERIODIC COMMIT
# LOAD CSV WITH HEADERS FROM 'https://s3-eu-west-1.amazonaws.com/football-transfers.neo4j.com/transfers-all.csv' AS row
# MATCH (player:Player {id: row.playerUri})
# MATCH (source:Club {id: row.sellerClubUri})
# MATCH (destination:Club {id: row.buyerClubUri})
# MERGE (t:Transfer {id: row.transferUri})
# ON CREATE SET t.season = row.season,
# t.fee = row.transferFee,
# t.timestamp = toInteger(row.timestamp)
# MERGE (t)-[ofPlayer:OF_PLAYER]->(player) SET ofPlayer.age = row.playerAge
# MERGE (t)-[:FROM_CLUB]->(source)
# MERGE (t)-[:TO_CLUB]->(destination)
# """)


# # Data Cleaning

# # Clean transfer fees
# graph.run("""
# MATCH (t:Transfer)
# WHERE t.fee contains "?" or t.fee contains "-"
# REMOVE t:Transfer
# SET t:TransferWithoutFee
# """)

# # Tag loan transfers
# graph.run("""
# MATCH (t:Transfer)
# WHERE t.fee STARTS WITH 'Loan'
# SET t:Loan
# """)

# # Convert text fees to numeric values
# graph.run("""
# MATCH (t:Transfer)
# WITH t, replace(replace(replace(replace(t.fee, "k", ""), "m", ""), "Loan fee:", ""), "£", "") AS rawNumeric
# WITH t,
# CASE
# WHEN t.fee ENDS WITH "k" THEN toFloat(apoc.number.exact.mul(trim(rawNumeric),"1000"))
# WHEN trim(t.fee) IN ["Free transfer", "ablösefrei ", "gratuito", "free", "free transfer", "Ablösefrei", "transfervrij"] THEN 0
# WHEN NOT(exists(t.fee)) THEN 0
# WHEN rawNumeric = '' THEN 0
# WHEN t.fee ENDS WITH "m" THEN toFloat(apoc.number.exact.mul(trim(rawNumeric),"1000000"))
# ELSE toFloat(trim(rawNumeric))
# END AS numericFee
# SET t.numericFee = numericFee
# """)

# # Create NEXT relationships between sequential transfers
# graph.run("""
# MATCH (p:Player)<-[:OF_PLAYER]-(transfer)
# WHERE transfer.numericFee > 0
# WITH p, transfer
# ORDER BY p.name, transfer.timestamp
# WITH p, collect(transfer) AS transfers
# WHERE size(transfers) > 1
# UNWIND range(0, size(transfers)-2) AS idx
# WITH transfers[idx] AS t1, transfers[idx+1] AS t2
# MERGE (t1)-[:NEXT]->(t2)
# """)

# # Create CASH_FLOW relationships for aggregated transfers
# graph.run("""
# MATCH (t:Transfer)
# WITH DISTINCT t.season AS season
# MATCH (seller)<-[:FROM_CLUB]-(t:Transfer)-[:TO_CLUB]->(buyer)
# WHERE t.season = season AND t.numericFee > 0
# WITH season, seller, buyer, sum(t.numericFee) AS cash_flow, count(t) AS player_count
# MERGE (buyer)-[:CASH_FLOW {total: cash_flow, playerCount: player_count, season: season}]->(seller)
# """)


# # Exploratory Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Count nodes by label
node_counts = graph.run("""
MATCH (n)
RETURN labels(n) AS NodeType, count(*) AS Count
ORDER BY Count DESC
""").to_data_frame()

# Count relationships by type
rel_counts = graph.run("""
MATCH ()-[r]->()
RETURN type(r) AS RelationshipType, count(*) AS Count
ORDER BY Count DESC
""").to_data_frame()

# Overall database statistics
db_stats = graph.run("""
CALL apoc.meta.stats()
YIELD nodeCount, relCount, labels, relTypes
RETURN nodeCount, relCount, labels, relTypes
""").to_data_frame()

# Player-Transfer relationship analysis
player_transfers = graph.run("""
MATCH (p:Player)<-[:OF_PLAYER]-(t:Transfer)
RETURN p.name AS Player, count(t) AS TransferCount
ORDER BY TransferCount DESC
LIMIT 10
""").to_data_frame()

# Club transfer network analysis
club_network = graph.run("""
MATCH (from:Club)<-[:FROM_CLUB]-(t:Transfer)-[:TO_CLUB]->(to:Club)
WHERE exists(t.numericFee) AND t.numericFee > 0
RETURN from.name AS SellingClub, to.name AS BuyingClub, 
       count(*) AS TransferCount, 
       sum(t.numericFee) AS TotalValue
ORDER BY TotalValue DESC
LIMIT 15
""").to_data_frame()

# Country transfer patterns
country_transfers = graph.run("""
MATCH (c1:Country)<-[:PART_OF]-(club1:Club)<-[:FROM_CLUB]-(t:Transfer)-[:TO_CLUB]->(club2:Club)-[:PART_OF]->(c2:Country)
WHERE c1 != c2 AND exists(t.numericFee) AND t.numericFee > 0
RETURN c1.name AS SourceCountry, c2.name AS DestinationCountry, 
       count(*) AS TransferCount, 
       sum(t.numericFee) AS TotalValue
ORDER BY TotalValue DESC
LIMIT 15
""").to_data_frame()


# Top 10 transfers by fee
top_transfers = graph.run("""
MATCH (transfer:Transfer)-[:OF_PLAYER]->(player),
      (from)<-[:FROM_CLUB]-(transfer)-[:TO_CLUB]->(to)
RETURN player.name, from.name, to.name, transfer.numericFee, transfer.season
ORDER BY transfer.numericFee DESC
LIMIT 10
""").to_data_frame()

# Transfers by season
transfers_by_season = graph.run("""
MATCH (t:Transfer)
WHERE exists(t.numericFee)
RETURN t.season AS season, count(t) AS count, sum(t.numericFee) AS totalValue
ORDER BY season
""").to_data_frame()

# Top clubs by transfer spending
top_spenders = graph.run("""
MATCH (buyer:Club)-[f:CASH_FLOW]->()
RETURN buyer.name AS club, sum(f.total) AS totalSpent
ORDER BY totalSpent DESC
LIMIT 10
""").to_data_frame()

# print('top_spenders===>', top_spenders)

# # Players with most transfers
most_transferred = graph.run("""
MATCH (p:Player)<-[:OF_PLAYER]-(t:Transfer)
WITH p, count(t) AS transferCount
WHERE transferCount > 2
RETURN p.name, transferCount
ORDER BY transferCount DESC
LIMIT 10
""").to_data_frame()


# # # Visualize top transfers
# plt.figure(figsize=(12, 6))
# sns.barplot(x='player.name', y='transfer.numericFee', data=top_transfers)
# plt.xticks(rotation=45, ha='right')
# plt.title('Top 10 Transfers by Fee')
# plt.tight_layout()
# plt.show()

# # # Visualize transfer market growth
# plt.figure(figsize=(14, 7))
# plt.plot(transfers_by_season['season'], transfers_by_season['totalValue'], marker='o')
# plt.title('Total Transfer Market Value by Season')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# Modeling
# PageRank to identify influential clubs in the transfer network
graph.run("""
CALL gds.graph.exists('footballTransferNetwork') YIELD exists
WITH exists
WHERE exists
CALL gds.graph.drop('footballTransferNetwork') YIELD graphName
RETURN graphName
""")

graph.run("""
CALL gds.graph.create('footballTransferNetwork',
  'Club',
  'CASH_FLOW',
  {
    relationshipProperties: ['total']
  }
)
""")

# Then run PageRank using your preferred format
graph.run("""
CALL gds.pageRank.write('footballTransferNetwork', {
  maxIterations: 20,
  dampingFactor: 0.85,
  relationshipWeightProperty: 'total',
  writeProperty: 'pagerank'
})
YIELD nodePropertiesWritten, ranIterations, didConverge
RETURN nodePropertiesWritten, ranIterations, didConverge
""")

# Retrieve top clubs by PageRank score
influential_clubs = graph.run("""
MATCH (c:Club)
WHERE exists(c.pagerank)
RETURN c.name AS club, c.pagerank AS influence
ORDER BY influence DESC
LIMIT 15
""").to_data_frame()

print("Top 15 Most Influential Clubs in Transfer Network:")
print(influential_clubs)

# Visualize top influential clubs
plt.figure(figsize=(12, 8))
sns.barplot(x='influence', y='club', data=influential_clubs)
plt.title('Most Influential Clubs in Transfer Network by PageRank')
plt.xlabel('PageRank Score')
plt.tight_layout()
plt.show()

# Community Detection Algorithms

# 1. Louvain Method
print("\n=== Running Louvain Community Detection ===")
graph.run("""
CALL gds.louvain.write('footballTransferNetwork', {
  relationshipWeightProperty: 'total',
  writeProperty: 'louvainCommunity',
  includeIntermediateCommunities: false
})
YIELD communityCount, modularity, modularities
RETURN communityCount, modularity, modularities
""")

# Get results from Louvain
louvain_communities = graph.run("""
MATCH (c:Club)
WHERE exists(c.louvainCommunity)
RETURN c.louvainCommunity AS community, count(*) AS clubCount
ORDER BY clubCount DESC
LIMIT 10
""").to_data_frame()

print("Louvain community sizes:")
print(louvain_communities)

# Get top clubs in each major community
top_clubs_by_community = graph.run("""
MATCH (c:Club)
WHERE exists(c.louvainCommunity)
WITH c ORDER BY c.pagerank DESC
WITH c.louvainCommunity AS community, collect(c.name)[..5] AS topClubs
WHERE size(topClubs) > 0
RETURN community, topClubs
ORDER BY community
LIMIT 10
""").to_data_frame()

print("\nTop clubs in major communities:")
print(top_clubs_by_community)

# 2. Label Propagation
print("\n=== Running Label Propagation Algorithm ===")
graph.run("""
CALL gds.labelPropagation.write('footballTransferNetwork', {
  relationshipWeightProperty: 'total',
  writeProperty: 'lpaComm'
})
YIELD communityCount, didConverge, ranIterations
RETURN communityCount, didConverge, ranIterations
""")

# Get results from LPA
lpa_communities = graph.run("""
MATCH (c:Club)
WHERE exists(c.lpaComm)
RETURN c.lpaComm AS community, count(*) AS clubCount
ORDER BY clubCount DESC
LIMIT 10
""").to_data_frame()

print("Label Propagation community sizes:")
print(lpa_communities)

# 3. Modularity Optimization with Leiden algorithm
print("\n=== Running Leiden Community Detection ===")
graph.run("""
CALL gds.leiden.write('footballTransferNetwork', {
  relationshipWeightProperty: 'total',
  writeProperty: 'leidenCommunity',
  maxIterations: 10
})
YIELD communityCount, modularity, modularities
RETURN communityCount, modularity, modularities
""")

# Get results from Leiden
leiden_communities = graph.run("""
MATCH (c:Club)
WHERE exists(c.leidenCommunity)
RETURN c.leidenCommunity AS community, count(*) AS clubCount
ORDER BY clubCount DESC
LIMIT 10
""").to_data_frame()

print("Leiden community sizes:")
print(leiden_communities)

# Visualize community sizes - Compare algorithms
plt.figure(figsize=(14, 7))

# Create a subplot for Louvain
plt.subplot(1, 3, 1)
sns.barplot(x='community', y='clubCount', data=louvain_communities[:5])
plt.title('Top 5 Louvain Communities')
plt.xlabel('Community ID')
plt.ylabel('Number of Clubs')

# Create a subplot for LPA
plt.subplot(1, 3, 2)
sns.barplot(x='community', y='clubCount', data=lpa_communities[:5])
plt.title('Top 5 Label Propagation Communities')
plt.xlabel('Community ID')
plt.ylabel('')

# Create a subplot for Leiden
plt.subplot(1, 3, 3)
sns.barplot(x='community', y='clubCount', data=leiden_communities[:5])
plt.title('Top 5 Leiden Communities')
plt.xlabel('Community ID')
plt.ylabel('')

plt.tight_layout()
plt.savefig('community_comparison.png')
plt.show()

# Analyze transfer patterns between communities
community_transfers = graph.run("""
MATCH (c1:Club)-[cf:CASH_FLOW]->(c2:Club)
WHERE exists(c1.louvainCommunity) AND exists(c2.louvainCommunity)
RETURN c1.louvainCommunity AS sourceCommunity,
       c2.louvainCommunity AS targetCommunity,
       count(cf) AS transferCount,
       sum(cf.total) AS totalValue
ORDER BY totalValue DESC
LIMIT 15
""").to_data_frame()

print("\nTop transfers between communities:")
print(community_transfers)

# Create a heatmap of inter-community transfers
pivot_transfers = community_transfers.pivot_table(
    index='sourceCommunity', 
    columns='targetCommunity', 
    values='totalValue',
    fill_value=0
)

plt.figure(figsize=(12, 10))
sns.heatmap(pivot_transfers, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Transfer Value Flow Between Communities (£)')
plt.tight_layout()
plt.savefig('community_transfers_heatmap.png')
plt.show()

# Analyze relationship between PageRank and transfer activity
pagerank_vs_activity = graph.run("""
MATCH (c:Club)
WHERE exists(c.pagerank)
OPTIONAL MATCH (c)<-[in:CASH_FLOW]-()
OPTIONAL MATCH (c)-[out:CASH_FLOW]->()
WITH c,
     count(distinct in) AS inDegree,
     count(distinct out) AS outDegree,
     sum(in.total) AS incomingValue,
     sum(out.total) AS outgoingValue
RETURN c.name AS club,
       c.pagerank AS pagerank,
       inDegree,
       outDegree,
       inDegree + outDegree AS totalConnections,
       CASE WHEN incomingValue IS NULL THEN 0 ELSE incomingValue END AS incomingValue,
       CASE WHEN outgoingValue IS NULL THEN 0 ELSE outgoingValue END AS outgoingValue
ORDER BY pagerank DESC
LIMIT 20
""").to_data_frame()

# # Profit Prediction Model

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score

# # Get features for profit prediction
# transfer_data = graph.run("""
# MATCH (p:Player)<-[:OF_PLAYER]-(t1)-[:NEXT]->(t2),
#       (club0)<-[:FROM_CLUB]-(t1)-[:TO_CLUB]->(club1)<-[:FROM_CLUB]-(t2)-[:TO_CLUB]->(club2)
# WHERE none(t in [t1, t2] where t:Loan)
# WITH p.name AS player, p.position AS position, 
#      SUBSTRING(t1.ofPlayer.age, 0, 2) AS buyAge,
#      SUBSTRING(t2.ofPlayer.age, 0, 2) AS sellAge,
#      t1.numericFee AS buyFee, t2.numericFee AS sellFee,
#      (t2.timestamp - t1.timestamp) / 60 / 60 / 24 / 365 AS yearsAtClub,
#      t2.numericFee - t1.numericFee AS profit
# RETURN player, position, toInteger(buyAge) AS buyAge, toInteger(sellAge) AS sellAge,
#        buyFee, sellFee, yearsAtClub, profit
# """).to_data_frame()

# # Prepare features and target
# X = transfer_data[['buyAge', 'sellAge', 'buyFee', 'yearsAtClub']]
# y = transfer_data['profit']

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Absolute Error: £{mae:,.2f}")
# print(f"R² Score: {r2:.2f}")

# # Feature importance
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': model.feature_importances_
# }).sort_values('Importance', ascending=False)

# # Feature importance visualization
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance)
# plt.title('Feature Importance for Transfer Profit Prediction')
# plt.tight_layout()
# plt.show()

# 2. Add a player career projection
graph.run("""
CALL gds.graph.exists('playerCareerNetwork') YIELD exists
WITH exists WHERE exists
CALL gds.graph.drop('playerCareerNetwork') YIELD graphName
RETURN graphName
""")

graph.run("""
CALL gds.graph.create('playerCareerNetwork',
  ['Player', 'Transfer'],
  {
    OF_PLAYER: { orientation: 'REVERSE' },
    NEXT: { orientation: 'NATURAL' }
  }
)
""")

# 3. Add a geographic talent flow network
graph.run("""
CALL gds.graph.exists('geoTalentNetwork') YIELD exists
WITH exists WHERE exists
CALL gds.graph.drop('geoTalentNetwork') YIELD graphName
RETURN graphName
""")

graph.run("""
CALL gds.graph.create('geoTalentNetwork',
  ['Country', 'Club'],
  {
    PART_OF: { orientation: 'REVERSE' }
  }
)
""")

# After your current PageRank analysis on footballTransferNetwork, add:

# Analyze player careers with path-finding algorithms
graph.run("""
CALL gds.betweenness.write('playerCareerNetwork', {
  writeProperty: 'careerCentrality'
})
YIELD nodePropertiesWritten
RETURN nodePropertiesWritten
""")

# Find key players in transfer chains
key_players = graph.run("""
MATCH (p:Player)
WHERE exists(p.careerCentrality)
RETURN p.name AS player, p.position AS position, p.careerCentrality AS centrality
ORDER BY centrality DESC
LIMIT 10
""").to_data_frame()

print("Key Players in Transfer Chains:")
print(key_players)

# Analyze country relationships with community detection
graph.run("""
CALL gds.louvain.write('geoTalentNetwork', {
  writeProperty: 'talentCluster'
})
YIELD communityCount, modularity
RETURN communityCount, modularity
""")

# Identify country talent clusters
country_clusters = graph.run("""
MATCH (c:Country)
WHERE exists(c.talentCluster)
RETURN c.talentCluster AS cluster, collect(c.name) AS countries
ORDER BY size(countries) DESC
LIMIT 5
""").to_data_frame()

print("Geographic Talent Clusters:")
print(country_clusters)

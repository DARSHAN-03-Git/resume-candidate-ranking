# Kubernetes API Demo

This demo deploys only the stateless API to a local kind cluster. PostgreSQL,
Neo4j, RabbitMQ, and Weaviate remain on Docker Compose, and the API pod reaches
their host-published ports through `host.docker.internal`.

A real Kubernetes migration of the stateful data services would require
PersistentVolumeClaims and StatefulSets. That work is out of scope for this
demo; the API is the stateless, appropriately Kubernetes-orchestrated piece.
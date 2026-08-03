# 0006: Introduce application services for workflow orchestration

- Status: Accepted
- Date: 2026-08-03

## Context

PDF ingestion, chunking, embedding, and vector search are independent capabilities.
User interfaces need to combine them into complete workflows without absorbing business logic or depending on implementation details.
The same workflows should later be reusable from a CLI, web API, or graphical interface.

## Decision

Introduce an `application` package containing thin orchestration services.
`DocumentSearchService` coordinates document chunking, retrieval indexing, and semantic search through narrow protocols.
The CLI remains responsible for argument parsing, dependency construction, and output serialization.

## Consequences

- Workflow logic can be reused by future interfaces.
- Domain services remain independently testable.
- Application services add another architectural layer and must remain thin.
- Provider construction currently stays at the CLI boundary and may move to a dedicated composition module as the number of providers grows.


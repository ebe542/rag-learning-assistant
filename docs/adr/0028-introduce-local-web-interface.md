# 0028: Introduce a local web interface

## Status

Accepted

## Context

The first alpha release exposes the complete learning workflow through a
machine-readable command-line interface. This proves the application services
and storage boundaries, but it still requires users to discover commands,
manage identifiers, and interpret JSON. The next product phase needs an
interface suitable for people who want to manage a library and study without
operating a terminal.

The existing application services are independent of the CLI and can be reused
by another interface. A graphical interface must preserve local-first operation,
remain testable without a browser or loaded ML model, and avoid turning CLI
subprocesses into an integration layer.

Streamlit can produce a prototype quickly, but its script rerun and session-state
model would shape navigation and long-running learning workflows around the UI
framework. PySide6 provides a native desktop interface, but adds a large binary
dependency and a separate desktop deployment problem before the user workflow
has stabilized.

## Decision

Add an optional local web interface built with FastAPI and server-rendered Jinja
templates. Package templates, CSS, and any later browser assets with the Python
distribution; the application must not require a CDN or internet connection.
Keep browser scripting optional and progressively enhance ordinary HTML forms
only when a workflow needs it.

Expose the interface through `rag-learn gui`. By default, the command binds only
to `127.0.0.1`, chooses a documented local port, and opens the system browser.
The initial interface has no authentication and therefore must not offer a
remote-bind option. It must not enable permissive CORS. State-changing routes
added in later milestones must validate same-origin requests before they become
available.

Place the web adapter beside the CLI under `interfaces`. Route handlers resolve
configuration and call application services directly; they must not invoke or
parse CLI commands. Page-specific presentation models translate domain results
into labels and navigation data without adding business rules.

Install the web stack through a separate `gui` optional dependency so the core
CLI and release-package checks remain lightweight. Tests construct the FastAPI
application with fake service dependencies and exercise routes through its test
client. The first delivery is read-only: it starts locally and lists the
available learning packages without loading embedding or generation models.

## Consequences

- Users gain a browser-based interface without a separate frontend build chain.
- CLI and GUI reuse the same application services and stored-data contracts.
- Server-rendered pages remain usable offline and can be tested with pytest.
- The first GUI milestone cannot mutate library data, which keeps its local
  security boundary small while the interface foundation is established.
- Running the GUI requires optional web dependencies and a local HTTP server.
- Native installers and desktop integration remain possible later, after the
  user workflow and distribution requirements are better understood.

---
applyTo: '**'
---
If you necessitate to execute test scripts, prepend `uv run` to any command so as to use the local environment. E.g., `uv run python <script_to_run>.py`.
Test scripts are very welcomed and must be placed in `fedmoe_plots/tests/` directory.
If you need to write logs don't use f-string formatting for the messages but use % formatting instead, e.g., `log.info("Message %s", variable)`.

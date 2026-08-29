
"""
Configure Logfire, an OpenTelemetry included in Pydantic AI
"""
import os

from genericsuite_codegen.utilities.app_logger import log_debug

DEBUG = False


def configure_logfire():
    _ = DEBUG and log_debug(">> Configuring Logfire...")
    import logfire

    # Set the OTEL_EXPORTER_OTLP_ENDPOINT environment variable to the URL of
    # your OpenTelemetry backend. If you're using a backend that requires
    # authentication, you may need to set other environment variables.
    # Of course, these can also be set outside the process, e.g. with
    #    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'

    # We configure Logfire to disable sending data to the Logfire OTel backend
    # itself. If you removed send_to_logfire=False, data would be sent to both
    # Logfire and your OpenTelemetry backend.
    logfire.configure(
        service_name='gscodegen',
        send_to_logfire=False
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)

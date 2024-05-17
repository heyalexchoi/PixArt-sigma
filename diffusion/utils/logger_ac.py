"""
Sets up logging. if SHOULD_LOG_TO_GCLOUD is set to 'true' in .env,
logs will be sent to google cloud. Can be imported in any script / app entrypoint.
Module level logging can still use python logging logger like:
```
import logging
logger = logging.getLogger(__name__)
```
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up basic configuration, with a logging level of INFO
# Set the logging level and format
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')

logger = logging.getLogger(__name__)

getLogger = logging.getLogger
get_logger = logging.getLogger

SHOULD_LOG_TO_GCLOUD = os.environ.get('SHOULD_LOG_TO_GCLOUD', False)

if SHOULD_LOG_TO_GCLOUD == 'true':
    import google.cloud.logging
    # Instantiates a client
    client = google.cloud.logging.Client()
    # Retrieves a Cloud Logging handler based on the environment
    # you're running in and integrates the handler with the
    # Python logging module. By default this captures all logs
    # at INFO level and higher
    client.setup_logging()
    logger.info("google cloud logging set up")

"""Main entry point for the refactored schema agent."""
import logging
from sys import argv

from src import JsonSchemaRepository, SchemaService, SchemaAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    """Run the schema agent with clean architecture."""
    logger.info("Starting Schema Agent application")

    # Initialize layers
    logger.info("Initializing repository, service, and agent layers")
    repository = JsonSchemaRepository("combined.json")
    service = SchemaService(repository)
    agent = SchemaAgent(service)
    logger.info("All layers initialized successfully")

    while True:
        user_input = input("You: ").strip() if len(argv) == 1 else argv[1]

        if user_input.lower() == 'exit':
            logger.info("User requested exit, shutting down")
            break

        if user_input.lower() == 'reset':
            logger.info("User requested conversation reset")
            agent.reset()
            continue

        if not user_input:
            continue

        logger.info(f"Processing user query: {user_input[:50]}...")
        response = agent.run(user_input)
        print(response)
        break


if __name__ == "__main__":
    main()

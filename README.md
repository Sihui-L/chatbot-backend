# fastapi-template

A template project for building applications using FastAPI with Docker. This setup provides a streamlined development environment and easy dependency management.


## Quickstart
1. **Build Docker image**: Run `make build` to create the Docker images for the FastAPI application and any supplementary services.
2. **Compile dependencies**: Run `make deps` to generate `requirements.txt` and `requirements.dev.txt` using piptools.
3. **Set up .env**: Create .env file
3. **Start containers and enter bash session**: Run `make start` to start the service containers and open an interactive bash shell in the main application container.
4. **Run app**: Run `make app` inside the container.

## Building the Service Containers
Running `make build` executes `docker-compose -f docker-compose.dev.yaml build`. This command builds a Docker image from `Dockerfile` for the FastAPI application and Docker images for other necessary services as specified in `docker-compose.dev.yaml`. The Dockerfile includes steps to:
- Create a user named `convergence_user`
- Install Python dependencies
- Update and install system packages
- Expose port 8081
- Start the FastAPI application on port 8081

## Compiling Dependencies
`piptools` helps manage dependencies by creating a `requirements.txt` file with all necessary packages and their compatible versions. Running `make deps` inside the container generates:
- `requirements.txt`: Contains packages required for running the application in production.
- `requirements.dev.txt`: Includes all packages from `requirements.txt` plus additional packages needed for development (e.g., testing tools).

## Starting the Service
Running `make start` executes the following commands:
- `docker-compose -f docker-compose.dev.yaml up -d`: Starts all the service containers in detached mode.
- `docker exec -it browser_agent bash`: Opens an interactive bash shell inside the `browser_agent` container for direct interaction.

This setup allows you to run and interact with the application in a development environment.


## Running tests
Our testing strategy prioritises integration tests to ensure robust API interactions, while maintaining the ability to run some unit tests if required. To run tests, use the `make tests` command inside the Docker container. This loads the environment from `tests/test.env`, executes tests in both `tests/unit` and `tests/integration` directories, and generates a coverage report. New tests can be added to the appropriate directory under `tests/`.

## Adding Resources
Resources in this project are managed using a custom utility system. Here's how it works:

Define resources by inheriting from `src.utils.util_manager.Util`:
```python
from abc import ABC, abstractmethod

class Util(ABC):
    @abstractmethod
    async def init(self):
        # Initialise and return the resource
        ...

    @abstractmethod
    async def cleanup(self, resource):
        # Clean up the resource (returned by self.init())
        ...
```

Resources are initalised at application startup and cleaned up at shutdown using FastAPI's [lifespan events](https://fastapi.tiangolo.com/advanced/events/).
Add resources to the UtilManager for type hinting and easy access:

```python
class Resources(UtilManager):
    @property
    def database(self) -> sqlite3.Connection:
        return self.initialised_resources.get(SQLite)

resources = Resources([SQLite])
```

Use resources in your application:

```python
from src.utils.utils import resources

resources.database.cursor()
```
This system allows for organised resource management and provides auto-completion via type hints throughout the application.


## Local Development
Docker Compose binds the volume of the container to the local volume. This setup allows you to make changes to the source code on your local machine, which are automatically reflected in the container without needing to rebuild it. Rebuilding the container is only necessary when adding or updating dependencies. Adding environment variables to `.env` requires restarting the container.

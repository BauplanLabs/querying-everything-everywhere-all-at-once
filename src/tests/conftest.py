import os
import uuid

import bauplan
import pytest
from dotenv import load_dotenv


# Load .env from project root
load_dotenv()
TEST_PREFIX = "qtm_test"  # prefix for all test branches/tables/tags
# Namespace to use for testing — can be overridden by BAUPLAN_TEST_NAMESPACE in .env or env vars
NAMESPACE = os.environ.get("BAUPLAN_TEST_NAMESPACE", "apo_test_multiverse")


@pytest.fixture(scope="session")
def s3_path():
    """S3 path used to create test tables.

    Set via BAUPLAN_TEST_S3_PATH in .env at the project root,
    or pass it as an environment variable when running pytest:

        BAUPLAN_TEST_S3_PATH=s3://bucket/path.parquet uv run pytest src/tests/ -v
    """
    path = os.environ.get("BAUPLAN_TEST_S3_PATH")
    if not path:
        pytest.skip("BAUPLAN_TEST_S3_PATH not set — add it to .env or pass as env var")
    return path


@pytest.fixture(scope="session")
def client():
    return bauplan.Client()


@pytest.fixture(scope="session", autouse=True)
def check_namespace(client):
    """Fail fast if the configured namespace doesn't exist on main."""
    assert client.has_namespace(NAMESPACE, ref="main"), (
        f"Namespace '{NAMESPACE}' does not exist on main — create it first"
    )


@pytest.fixture()
def unique_id():
    """Short unique suffix for naming test resources."""
    return uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def username(client):
    """Fetch the authenticated username from the Bauplan service."""
    try:
        return client.info().user.username
    except Exception as e:
        pytest.skip(f"Cannot fetch username from Bauplan: {e}")


@pytest.fixture()
def test_branch(client, unique_id, username):
    """Create a disposable branch from main, clean up after the test."""
    name = f"{username}.{TEST_PREFIX}_{unique_id}"
    client.create_branch(name, from_ref="main")
    yield name
    client.delete_branch(name, if_exists=True)

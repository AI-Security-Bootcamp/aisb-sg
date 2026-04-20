# %%

# Ensure the root directory is in the path for imports
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Common imports
import requests

print("It works!")


# %%

from day0_test import test_prerequisites

# Run the prerequisite checks
test_prerequisites()


# %%
from dataclasses import dataclass


@dataclass
class UserIntel:
    username: str
    name: str | None
    location: str | None
    email: str | None
    repo_names: list[str]


def analyze_user_behavior(username: str = "karpathy") -> UserIntel:
    """
    Analyze a user's GitHub activity patterns.
    This is the kind of profiling attackers might do for social engineering.

    Returns:
        The user's name, location, email, and 5 most recently updated repos.
    """
    # TODO: Return information about the given GitHub user
    # 1. Make a GET request to: https://api.github.com/users/{username}
    # 2. Extract name, location, and email from the response
    # 3. Make another GET request to: https://api.github.com/users/{username}/repos?sort=updated&per_page=5
    # 4. Extract repository names (limit to 5)
    # 5. Return a UserIntel object with the gathered information
    # Get user info
    user_response = requests.get(f"https://api.github.com/users/{username}")
    if user_response.status_code != 200:
        # Return empty intel if user not found
        return UserIntel(username=username, name=None, location=None, email=None, repo_names=[])

    user_data = user_response.json()

    # Get user's repositories (sorted by most recently updated)
    repos_response = requests.get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5")
    repo_names = []
    if repos_response.status_code == 200:
        repos = repos_response.json()
        repo_names = [repo["name"] for repo in repos[:5]]  # Limit to 5 repos

    return UserIntel(
        username=username,
        name=user_data.get("name"),
        location=user_data.get("location"),
        email=user_data.get("email"),
        repo_names=repo_names,
    )
    pass


# %%
from day0_test import test_analyze_user_behavior

test_analyze_user_behavior(analyze_user_behavior)

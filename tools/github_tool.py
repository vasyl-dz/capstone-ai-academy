"""
GitHub Repository Analysis Tool
Uses PyGithub to fetch and analyze GitHub repositories
"""

from github import Github, GithubException
from typing import Optional, Dict
import os


class GitHubTool:
    """Tool for analyzing GitHub repositories."""
    
    def __init__(self, access_token: Optional[str] = None):
        """Initialize GitHub tool with optional access token."""
        token = access_token or os.getenv("GITHUB_TOKEN")
        self.github = Github(token) if token else Github()
    
    def analyze_repository(self, repo_url: str) -> Dict[str, str]:
        """
        Analyze a GitHub repository by fetching README and metadata.
        
        Args:
            repo_url: GitHub repository URL (e.g., https://github.com/owner/repo)
        
        Returns:
            Dictionary with repository information
        """
        try:
            # Extract owner and repo name from URL
            parts = repo_url.rstrip('/').split('/')
            if len(parts) < 2:
                return {"error": "Invalid GitHub URL format"}
            
            owner, repo_name = parts[-2], parts[-1]
            
            # Fetch repository
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # Get README content
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8')
            except GithubException:
                readme_content = "No README found"
            
            # Gather metadata
            result = {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description or "No description",
                "stars": str(repo.stargazers_count),
                "forks": str(repo.forks_count),
                "language": repo.language or "Not specified",
                "topics": ", ".join(repo.get_topics()) if repo.get_topics() else "None",
                "readme": readme_content[:3000],  # Limit README size
                "url": repo.html_url,
                "created_at": str(repo.created_at),
                "updated_at": str(repo.updated_at),
            }
            
            return result
            
        except GithubException as e:
            return {"error": f"GitHub API error: {str(e)}"}
        except Exception as e:
            return {"error": f"Error analyzing repository: {str(e)}"}
    
    def format_repo_info(self, repo_info: Dict[str, str]) -> str:
        """Format repository information for LLM context."""
        if "error" in repo_info:
            return f"Error: {repo_info['error']}"
        
        formatted = f"""Repository: {repo_info['full_name']}
Description: {repo_info['description']}
Language: {repo_info['language']}
Stars: {repo_info['stars']} | Forks: {repo_info['forks']}
Topics: {repo_info['topics']}
Created: {repo_info['created_at']}
Last Updated: {repo_info['updated_at']}

README:
{repo_info['readme']}
"""
        return formatted

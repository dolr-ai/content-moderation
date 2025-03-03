#!/usr/bin/env python3
"""
Prompt templates for content moderation system.
Uses Jinja2 for template rendering.
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import jinja2

# Define the base system prompt for content moderation
SYSTEM_PROMPT = """You are a content moderation expert. Your task is to analyze content and categorize it into one of the following categories:

1. hate_or_discrimination: Content targeting protected characteristics with negative intent/prejudice
2. violence_or_threats: Content that threatens, depicts, or promotes violence
3. offensive_language: Hostile or inappropriate content WITHOUT targeting protected characteristics
4. nsfw_content: Explicit sexual content or material intended to arouse
5. spam_or_scams: Deceptive or unsolicited content designed to mislead
6. clean: Content that is allowed and doesn't fall into above categories

Please format your response exactly as:
Category: [exact category_name]
Confidence: [HIGH/MEDIUM/LOW]
Explanation: [short 1/2 line explanation]"""

# Valid categories for content moderation
VALID_CATEGORIES = {
    "hate_or_discrimination",
    "violence_or_threats",
    "offensive_language",
    "nsfw_content",
    "spam_or_scams",
    "clean",
}


class PromptManager:
    """Manager for prompt templates using Jinja2."""

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            templates_dir: Directory containing prompt templates.
                          If None, uses the default 'prompts' directory.
        """
        if templates_dir is None:
            # Try to find the prompts directory
            current_file = Path(__file__).resolve()
            templates_dir = current_file.parent / "templates"

            if not templates_dir.exists():
                templates_dir = current_file.parent

        self.templates_dir = Path(templates_dir)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Create default templates if they don't exist
        self._ensure_default_templates()

    def _ensure_default_templates(self) -> None:
        """Create default templates if they don't exist."""
        self.templates_dir.mkdir(exist_ok=True, parents=True)

        # Create the base moderation template
        base_template_path = self.templates_dir / "moderation_base.j2"
        if not base_template_path.exists():
            with open(base_template_path, "w") as f:
                f.write(SYSTEM_PROMPT)

        # Create the RAG template with examples
        rag_template_path = self.templates_dir / "moderation_rag.j2"
        if not rag_template_path.exists():
            rag_template = """{{ system_prompt }}

Here are some similar examples to help you:

{% for example in examples %}
Example {{ loop.index }}:
Content: {{ example.text }}
Category: {{ example.category }}
{% if example.distance %}Similarity: {{ "%.2f"|format(1.0 - example.distance) }}{% endif %}
{% endfor %}

Now, please analyze the following content:
Content: {{ query }}"""

            with open(rag_template_path, "w") as f:
                f.write(rag_template)

    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a template with the given variables.

        Args:
            template_name: Name of the template file
            **kwargs: Variables to pass to the template

        Returns:
            The rendered template as a string
        """
        template = self.env.get_template(f"{template_name}.j2")
        return template.render(**kwargs)

    def get_moderation_prompt(self, query: str) -> str:
        """
        Get the basic moderation prompt for a query.

        Args:
            query: The text to moderate

        Returns:
            The complete prompt for the moderation task
        """
        return self.render_template("moderation_base", query=query)

    def get_rag_prompt(self, query: str, examples: List[Dict[str, Any]]) -> str:
        """
        Get the RAG-enhanced moderation prompt with examples.

        Args:
            query: The text to moderate
            examples: List of example documents with their categories

        Returns:
            The complete RAG prompt for the moderation task
        """
        return self.render_template(
            "moderation_rag",
            system_prompt=SYSTEM_PROMPT,
            query=query,
            examples=examples,
        )


# Create a singleton instance
prompt_manager = PromptManager()

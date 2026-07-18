from pathlib import Path

from utils.logger import get_logger

logger = get_logger("prompt_loader")
BASE_PROMPT_PATH = Path(__file__).parent.parent / "prompts"


def load_prompt_template(file_name: str) -> str:
    """Read prompt text from prompt directory"""
    target_file = BASE_PROMPT_PATH / file_name
    if not target_file.exists():
        logger.error(f"Prompt file missing: {file_name}")
        return ""
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content

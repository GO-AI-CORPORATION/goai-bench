"""Custom exceptions for GO AI Bench."""


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded from HuggingFace Hub or local path."""

    def __init__(self, model_id: str, reason: str = "") -> None:
        self.model_id = model_id
        self.reason = reason
        msg = f"Failed to load model '{model_id}'."
        if reason:
            msg += f" {reason}"
        msg += (
            " Suggestion: verify the model ID exists on https://huggingface.co/models"
            " and that you have the required dependencies installed."
        )
        super().__init__(msg)


class EmptyDatasetError(Exception):
    """Raised when evaluation data is empty after loading/filtering."""

    def __init__(self, language: str = "", task: str = "", domain: str = "") -> None:
        self.language = language
        self.task = task
        self.domain = domain
        msg = (
            f"No evaluation data found (language={language}, task={task},"
            f" domain={domain}). Populate data files under data/evaluation/"
            f" or specify a HuggingFace dataset with --dataset hf:<dataset_id>."
        )
        super().__init__(msg)

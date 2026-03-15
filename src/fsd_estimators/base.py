class BaseFsdEstimator:
    """
    Base class for FSD estimators. Subclasses should override hooks as needed.
    """
    def __init__(self, *args, **kwargs):
        pass

    def on_task_start(self, model, task_id, **kwargs):
        """Called at the start of each task."""
        pass

    def on_task_end(self, task_id, model, task_info, **kwargs):
        """Called at the end of each task."""
        pass

    def get_fsd(self, *args, **kwargs):
        """Compute the FSD value. Subclasses must implement this."""
        raise NotImplementedError 
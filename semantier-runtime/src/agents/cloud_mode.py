"""Semantier Runtime is always a managed composition, including local development."""
import os


def require_cloud_control(environment=None):
    environment = os.environ if environment is None else environment
    if environment.get("SEMANTIER_RUNTIME", "0") != "1":
        raise ValueError("Semantier control requires SEMANTIER_RUNTIME=1")


def require_managed_runtime(environment=None):
    environment = os.environ if environment is None else environment
    if environment.get("SEMANTIER_RUNTIME", "1") != "1":
        raise ValueError("Semantier Runtime requires SEMANTIER_RUNTIME=1")
    environment["SEMANTIER_RUNTIME"] = "1"

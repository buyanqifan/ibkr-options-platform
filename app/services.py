"""Global service registry for sharing services between callbacks."""

# Global service registry
_services: dict | None = None


def get_services() -> dict | None:
    """Access shared service instances from any callback."""
    return _services


def set_services(services: dict) -> None:
    """Initialize the global service registry."""
    global _services
    _services = services
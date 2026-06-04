"""Security extension point for Viewer API dependencies.

The Viewer API is currently a local development service and does not authenticate
requests. Keep request safety checks in route services and path validators until
a concrete authentication or authorization adapter is introduced.
"""


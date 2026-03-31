"""Entry point: python -m alpha_lab.dashboard.api"""

import uvicorn

from alpha_lab.dashboard.api.server import create_app

uvicorn.run(create_app(), host="0.0.0.0", port=8000)

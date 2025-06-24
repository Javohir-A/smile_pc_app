# debug_ucode_responses.py - Debug UCode API response structures
import sys
import os
import logging
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ucode_sdk.sdk import new, Config

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import time
app_id = "P-QlWnuJCdfy32dsQoIjXHQNScO7DR2TdL"
base_url = "https://api.client.u-code.io"
ucode_api = new(Config(app_id, base_url, __name__))

human_api_data = {
    "full_name": f"Unknown_Client_{int(time.time())}",
    # "company_id": "",  # Add if needed
    # "branch_id": "",   # Add if needed
    "photos": ["https://cdn.u-code.io/efba2b71-f75f-482f-9b4e-6538961864b7/Media/6dd30947-615a-4a71-af7b-6c347d96a576_unknown_face_001_1750803100_quality_1.06.jpg"],
    "type": ["client"]
}

res = ucode_api.items("human").create(human_api_data).exec()

print(res)

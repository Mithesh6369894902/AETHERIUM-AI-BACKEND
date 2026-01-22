from fastapi import Header, HTTPException

API_KEY = "AETHERIUM_KEY"   # Academic demo key

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid API Key"
        )

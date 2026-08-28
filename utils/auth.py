import subprocess
import base64
import sys
import os
import json
import time

_cached_ac_token = None
_cached_insight_token = None

CACHE_FILE = os.path.join(os.environ.get('AGENT_PROJECT_ROOT', os.getcwd()), 'Result', 'temp', 'token_cache.json')

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                # Check expiration (e.g. 25 minutes = 1500 seconds)
                if time.time() - data.get('timestamp', 0) < 1500:
                    return data
        except:
            pass
    return {}

def save_cache(ac_token, insight_token):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    data = {'timestamp': time.time()}
    if ac_token: data['ac_token'] = ac_token
    if insight_token: data['insight_token'] = insight_token
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f)

def get_appleconnect_token():
    global _cached_ac_token
    if _cached_ac_token: return _cached_ac_token
    
    cache = load_cache()
    if 'ac_token' in cache:
        _cached_ac_token = cache['ac_token']
        return _cached_ac_token
        
    print("正在通过 AppleConnect 获取 Floodgate Token...")
    try:
        result = subprocess.run([
            "appleconnect", "getToken", "-t", "oauth", "-G", "pkce",
            "-C", "hvys3fcwcteqrvw3qzkvtk86viuoqv",
            "-o", "openid,dsid,accountname,email,groups",
            "--interactivity-type", "none"
        ], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "oauth-id-token" in line or "id-token" in line or "oauth-id" in line:
                parts = line.replace(":", " ").split()
                for i, part in enumerate(parts):
                    if part in ("oauth-id-token", "id-token", "oauth-id") and i + 1 < len(parts):
                        _cached_ac_token = parts[i+1].strip()
                        save_cache(_cached_ac_token, _cached_insight_token)
                        return _cached_ac_token
    except Exception:
        pass
    sys.exit(1)

def get_insight_token() -> str:
    global _cached_insight_token
    if _cached_insight_token: return _cached_insight_token
    
    cache = load_cache()
    if 'insight_token' in cache:
        _cached_insight_token = cache['insight_token']
        return _cached_insight_token
        
    try:
        token_bytes = subprocess.check_output([
            "appleconnect", "getToken", 
            "-I", "176361", 
            "-t", "default", 
            "-E", "prod", 
            "-i", "touchId", 
            "-n", "gui"
        ])
        _cached_insight_token = base64.b64decode(token_bytes).decode('utf-8').strip()
        save_cache(_cached_ac_token, _cached_insight_token)
        return _cached_insight_token
    except subprocess.CalledProcessError as e:
        print(f"❌ 获取 Insight Token 失败: {e}")
        raise

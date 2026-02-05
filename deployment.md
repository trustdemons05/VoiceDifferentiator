# 🌍 Deployment Guide: Exposing Your API

Since you have the code running locally on your machine, the fastest way to make it accessible to the world (and link it to your domain) is using a **Tunnel**.

## Option A: Cloudflare Tunnel (Recommended for Custom Domains)
This is free, secure, and allows you to use your own domain (e.g., `api.yourdomain.com`).

### 1. Install Cloudflare Tunnel (`cloudflared`)
Run this in PowerShell to download the verified Windows executable:
```powershell
# Create a folder for tools
mkdir c:\tools
cd c:\tools

# Download cloudflared
Invoke-WebRequest -Uri https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile cloudflared.exe
```

### 2. Login to Cloudflare
This connects your machine to your Cloudflare account (where your domain is managed).
```powershell
.\cloudflared.exe tunnel login
```
*   A browser window will open. Select the domain you want to use.

### 3. Create a Tunnel
```powershell
.\cloudflared.exe tunnel create voice-api
```
*   Copy the **Tunnel ID** (a long UUID like `d4c3b2a1-...`) from the output.

### 4. Route Your Domain to Localhost
Replace `<UUID>` with your Tunnel ID and `api.yourdomain.com` with your desired domain.
```powershell
# 1. Configure the tunnel target
# Create a config.yml file
echo "url: http://localhost:8000" > config.yml
echo "tunnel: <UUID>" >> config.yml
echo "credentials-file: C:\Users\baksh\.cloudflared\<UUID>.json" >> config.yml

# 2. Assign the domain DNS
.\cloudflared.exe tunnel route dns voice-api api.yourdomain.com
```

### 5. Run it!
```powershell
.\cloudflared.exe tunnel run voice-api
```
**Success!** Your local server (`localhost:8000`) is now live at `https://api.yourdomain.com`.

---

## Option B: Ngrok (Fastest, Random URL)
If you don't use Cloudflare for DNS, or just want a quick link:

1.  **Download Ngrok**: [https://ngrok.com/download](https://ngrok.com/download)
2.  **Run**:
    ```powershell
    ngrok http 8000
    ```
3.  **Result**: It will give you a URL like `https://a1b2-c3d4.ngrok-free.app`. Use this in the endpoint tester.

---

## 🔒 Important: Update Test Scripts
Once you have your public URL, test it using the verify script:

```python
# Update verify_tester_config.py
API_URL = "https://api.yourdomain.com/api/v1/detect" 
```

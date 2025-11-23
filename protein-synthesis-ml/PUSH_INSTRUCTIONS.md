# How to Push to GitHub - Simple Steps

## The Problem
Git needs to authenticate you to push code. Since SSH isn't set up, we'll use HTTPS with a Personal Access Token.

## Step-by-Step Instructions

### Step 1: Create a Personal Access Token on GitHub

1. **Go to GitHub**: Open https://github.com/settings/tokens in your browser
2. **Click**: "Generate new token" → "Generate new token (classic)"
3. **Name it**: "Drug-Discovery-Push" (or any name you like)
4. **Select expiration**: Choose how long it should last (30 days, 90 days, or no expiration)
5. **Check the box**: `repo` (this gives full access to repositories)
6. **Scroll down and click**: "Generate token"
7. **IMPORTANT**: Copy the token immediately! It looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - You won't be able to see it again!

### Step 2: Push Your Code

Run this command in your terminal:

```bash
cd /Users/sanjayanasuri/protein-synthesis-ml
git push -u origin main
```

When it asks for:
- **Username**: Enter `sanjayanasuri`
- **Password**: Paste your Personal Access Token (NOT your GitHub password!)

### Step 3: Done!

After successful push, your code will be at:
**https://github.com/sanjayanasuri/Drug-Discovery**

---

## Alternative: Use GitHub Desktop

If you prefer a GUI:
1. Download GitHub Desktop: https://desktop.github.com/
2. Sign in with your GitHub account
3. Add the repository
4. Click "Publish repository"

---

## Troubleshooting

**If it says "Authentication failed":**
- Make sure you're using the token, not your password
- Make sure the token has `repo` permission checked
- Try generating a new token

**If it says "Repository not found":**
- Make sure the repository exists at: https://github.com/sanjayanasuri/Drug-Discovery
- Make sure you're logged into the correct GitHub account


@echo off
REM deploy.bat - Windows deployment script for Crop Disease Prediction System

echo 🚀 Deploying Crop Disease Prediction System to Vercel
echo ==================================================

REM Check if Vercel CLI is installed
vercel --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Vercel CLI not found. Installing...
    npm install -g vercel
)

REM Check if user is logged in to Vercel
vercel whoami >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔐 Please login to Vercel:
    vercel login
    if %errorlevel% neq 0 (
        echo ❌ Login failed. Please try again.
        pause
        exit /b 1
    )
)

REM Deploy to Vercel
echo 📦 Deploying to Vercel...
vercel --prod

REM Set environment variables
echo.
echo 🔧 Setting up environment variables...
echo Please set the following environment variables in your Vercel dashboard:
echo - SECRET_KEY: A secure random string
echo - GOOGLE_API_KEY: Your Google Gemini API key
echo - FLASK_ENV: production
echo.

echo ✅ Deployment complete!
echo 🌐 Your app will be available at the URL shown above
echo.
echo 📝 Next steps:
echo 1. Go to your Vercel dashboard
echo 2. Add the environment variables listed above
echo 3. Redeploy if needed
echo 4. Test your deployed application

pause
#!/bin/bash
# Setup script for LinkedIn Profile Scraper

echo "Setting up LinkedIn Profile Scraper..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a .env file with your API keys:"
echo "   SERPER_API_KEY=your_serper_api_key"
echo "   GROQ_API_KEY=your_groq_api_key"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the scraper:"
echo "   python linkedin_scraper.py"




# Email Insights Dashboard

A powerful local application that connects to Gmail, fetches customer emails, and analyzes them using AI to provide insights similar to those offered by a world-class team of consultants.

## Features

- **Gmail Integration**: Secure OAuth 2.0 authentication with Gmail API
- **AI-Powered Analysis**: 
  - Sentiment analysis (positive/negative/neutral)
  - Email classification (complaint, inquiry, feedback)
  - Topic extraction using TF-IDF and K-means clustering
- **Interactive Dashboard**: Beautiful web interface with real-time charts
- **Model Training**: Fine-tune DistilBERT model on your email data
- **Local Deployment**: Runs entirely on your machine for privacy

## Project Structure

```
email_insights/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── credentials.json          # Google OAuth credentials (you need to create this)
├── token.pickle             # OAuth token (generated after first run)
├── templates/
│   └── index.html           # Web dashboard template
├── saved_model/             # Directory for trained models
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_app.py
├── pytest.ini              # Test configuration
└── README.md               # This file
```

## Prerequisites

- Python 3.7 or higher
- Google Cloud Console account
- Gmail account with emails to analyze

## Installation

1. **Clone or download the project**
   ```bash
   cd email_insights
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google OAuth credentials**
   
   a. Go to the [Google Cloud Console](https://console.cloud.google.com/)
   
   b. Create a new project or select an existing one
   
   c. Enable the Gmail API:
      - Navigate to "APIs & Services" > "Library"
      - Search for "Gmail API" and enable it
   
   d. Create OAuth credentials:
      - Go to "APIs & Services" > "Credentials"
      - Click "Create Credentials" > "OAuth client ID"
      - Choose "Desktop app" as the application type
      - Give it a name (e.g., "Email Insights App")
      - Click "Create"
   
   e. Download the credentials file and rename it to `credentials.json`
   
   f. Place `credentials.json` in the project root directory

4. **Download NLTK data** (done automatically on first run)
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to `http://localhost:5000`

3. **First-time setup**:
   - Click "Fetch and Analyze Emails"
   - A browser window will open for Google OAuth authentication
   - Grant permissions to access your Gmail
   - The app will fetch and analyze your emails

4. **View insights**:
   - Sentiment distribution charts
   - Email type classification
   - Topic clustering
   - Detailed email data table

5. **Train custom model** (optional):
   - Click "Train Model" to fine-tune the AI model on your specific email data
   - This improves classification accuracy for your use case

## API Endpoints

- `GET /` - Main dashboard page
- `POST /fetch_and_analyze` - Fetch emails from Gmail and analyze them
- `POST /train_model` - Train the AI model on your email data

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest tests/test_app.py::TestTextProcessing  # Run specific test class
```

## Configuration

### Environment Variables

You can set these environment variables to customize behavior:

- `FLASK_ENV=development` - Enable debug mode
- `MAX_EMAILS=50` - Maximum number of emails to fetch (default: 10)

### Model Configuration

The application uses several pre-trained models:

- **Sentiment Analysis**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Email Classification**: `facebook/bart-large-mnli`
- **Topic Extraction**: TF-IDF + K-means clustering

## Troubleshooting

### Common Issues

1. **"No module named 'transformers'"**
   ```bash
   pip install transformers torch
   ```

2. **"Gmail API not enabled"**
   - Go to Google Cloud Console
   - Enable the Gmail API for your project

3. **"Invalid credentials"**
   - Check that `credentials.json` is in the project root
   - Verify the credentials file is valid JSON
   - Ensure the OAuth client is configured for "Desktop app"

4. **"No emails found"**
   - Check that your Gmail account has emails in the inbox
   - Verify OAuth permissions were granted
   - Try increasing `max_results` in the `fetch_emails` function

5. **Model loading errors**
   - Ensure you have internet connection for first-time model downloads
   - Check available disk space (models are ~500MB each)

### Performance Tips

- For large email volumes, consider processing in batches
- Use the training feature to improve classification accuracy
- Monitor memory usage with large datasets

## Security & Privacy

- All processing happens locally on your machine
- No email data is sent to external servers (except for model downloads)
- OAuth tokens are stored locally in `token.pickle`
- Credentials are stored in `credentials.json` (keep this secure)

## Customization

### Adding New Email Categories

Edit the `classify_email` function in `app.py`:

```python
def classify_email(text, candidate_labels=["complaint", "inquiry", "feedback", "support", "billing"]):
    # Your custom labels here
```

### Modifying Analysis Parameters

Adjust these parameters in `app.py`:

```python
# Topic extraction
def extract_topics(texts, n_clusters=5):  # Change number of clusters

# Email fetching
def fetch_emails(service, user_id='me', label_ids=['INBOX'], max_results=10):  # Change max results
```

### Styling the Dashboard

Edit `templates/index.html` to customize the appearance, colors, and layout.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the test suite for usage examples
3. Check the Google Gmail API documentation
4. Open an issue with detailed error messages

## Roadmap

- [ ] Support for multiple email accounts
- [ ] Advanced topic modeling (LDA)
- [ ] Email export functionality
- [ ] Real-time email monitoring
- [ ] Custom dashboard themes
- [ ] API rate limiting
- [ ] Docker containerization

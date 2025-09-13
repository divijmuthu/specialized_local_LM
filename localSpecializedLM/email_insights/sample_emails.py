"""
Sample email data for local testing and demonstration
"""

SAMPLE_EMAILS = [
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Complaint: Product not working as expected'},
                {'name': 'From', 'value': 'angry.customer@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 09:15:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Question about billing and pricing'},
                {'name': 'From', 'value': 'confused.user@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 10:30:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Thank you for excellent service!'},
                {'name': 'From', 'value': 'happy.customer@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 11:45:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Technical support needed urgently'},
                {'name': 'From', 'value': 'frustrated.user@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 14:20:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Feature request: Add dark mode'},
                {'name': 'From', 'value': 'suggestive.user@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 15:10:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Refund request - defective product'},
                {'name': 'From', 'value': 'disappointed.customer@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 16:30:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'How to integrate with our system?'},
                {'name': 'From', 'value': 'technical.lead@company.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 17:00:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Amazing product! Highly recommend'},
                {'name': 'From', 'value': 'satisfied.user@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 18:15:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Account locked - need help'},
                {'name': 'From', 'value': 'locked.user@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 19:30:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    },
    {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Feedback: Great customer service'},
                {'name': 'From', 'value': 'appreciative.customer@example.com'},
                {'name': 'Date', 'value': 'Mon, 12 Sep 2024 20:45:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
            }
        }
    }
]

# Realistic email body content for each email
EMAIL_BODIES = [
    "I am extremely disappointed with your product. It doesn't work as advertised and I've wasted hours trying to get it to function properly. This is unacceptable and I demand a full refund immediately.",
    
    "Hi, I'm confused about the billing structure. Can you please explain the different pricing tiers and what features are included in each? Also, when will I be charged for the next billing cycle?",
    
    "I just wanted to take a moment to thank you for the excellent service I received. Your support team was incredibly helpful and resolved my issue quickly. I'm very satisfied with your product!",
    
    "URGENT: I'm having a critical issue with the system and need immediate technical support. The application keeps crashing and I can't access my data. Please help as soon as possible.",
    
    "I love using your product! One suggestion I have is to add a dark mode feature. It would be really helpful for users who work late at night. Keep up the great work!",
    
    "I received a defective product and need to request a refund. The item arrived damaged and doesn't work at all. Please process my refund request as soon as possible.",
    
    "We're interested in integrating your API with our existing system. Could you provide documentation on how to connect and what endpoints are available? We need this for our enterprise solution.",
    
    "This product is absolutely fantastic! I've been using it for months and it has significantly improved my workflow. I've already recommended it to several colleagues. Thank you!",
    
    "My account seems to be locked and I can't log in. I've tried resetting my password but nothing works. Can you please help me regain access to my account?",
    
    "I wanted to provide feedback on the excellent customer service I received. The representative was knowledgeable, patient, and resolved my issue quickly. This is how customer service should be done!"
]

def get_sample_emails():
    """Get sample emails with realistic content"""
    emails = []
    for i, email in enumerate(SAMPLE_EMAILS):
        # Create a copy of the email structure
        email_copy = {
            'payload': {
                'headers': email['payload']['headers'].copy(),
                'body': {'data': email['payload']['body']['data']}
            }
        }
        emails.append(email_copy)
    return emails

def get_sample_email_bodies():
    """Get realistic email body content"""
    return EMAIL_BODIES

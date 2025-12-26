import qrcode
from io import BytesIO
from typing import Tuple
import base64
from datetime import datetime
import uuid


class QRCodeGenerator:
    """
    Generates QR codes for hospital feedback submissions
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
    
    def generate_feedback_qr(self, qr_type: str = "general", 
                            metadata: dict = None) -> Tuple[str, str]:
        """
        Generate QR code for feedback submission
        qr_type: 'general', 'doctor', 'food', 'room', 'complaint'
        Returns: (qr_code_base64, feedback_url)
        """
        metadata = metadata or {}
        qr_id = str(uuid.uuid4())
        
        # Build feedback URL
        feedback_url = f"{self.base_url}/feedback/{qr_type}?qr_id={qr_id}"
        
        if metadata:
            for key, value in metadata.items():
                feedback_url += f"&{key}={value}"
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(feedback_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}", feedback_url
    
    def generate_doctor_feedback_qr(self, doctor_id: str, doctor_name: str) -> Tuple[str, str]:
        """Generate QR code for doctor-specific feedback"""
        return self.generate_feedback_qr(
            "doctor",
            {"doctor_id": doctor_id, "doctor_name": doctor_name}
        )
    
    def generate_room_feedback_qr(self, room_id: str) -> Tuple[str, str]:
        """Generate QR code for room feedback"""
        return self.generate_feedback_qr(
            "room",
            {"room_id": room_id}
        )
    
    def generate_food_feedback_qr(self) -> Tuple[str, str]:
        """Generate QR code for food quality feedback"""
        return self.generate_feedback_qr("food")
    
    def generate_complaint_qr(self) -> Tuple[str, str]:
        """Generate QR code for complaint submission"""
        return self.generate_feedback_qr("complaint")


class FeedbackForm:
    """
    Manages feedback form generation and validation
    """
    
    @staticmethod
    def get_general_feedback_form() -> dict:
        """Get general hospital feedback form structure"""
        return {
            "form_id": "general_feedback",
            "title": "Hospital Experience Feedback",
            "sections": [
                {
                    "section": "Overall Experience",
                    "fields": [
                        {
                            "name": "overall_rating",
                            "label": "Overall Hospital Experience",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "visit_date",
                            "label": "Date of Visit",
                            "type": "date"
                        },
                        {
                            "name": "comments",
                            "label": "Please share your feedback",
                            "type": "textarea",
                            "max_length": 1000
                        }
                    ]
                }
            ],
            "languages": ["en", "es", "fr", "de", "it", "pt", "hi", "ar", "zh", "ja"]
        }
    
    @staticmethod
    def get_doctor_feedback_form() -> dict:
        """Get doctor-specific feedback form"""
        return {
            "form_id": "doctor_feedback",
            "title": "Doctor Feedback",
            "sections": [
                {
                    "section": "Doctor Performance",
                    "fields": [
                        {
                            "name": "doctor_name",
                            "label": "Doctor Name",
                            "type": "text",
                            "readonly": True
                        },
                        {
                            "name": "communication",
                            "label": "Communication Skills",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "expertise",
                            "label": "Medical Expertise",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "compassion",
                            "label": "Compassion & Empathy",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "waiting_time",
                            "label": "Waiting Time",
                            "type": "radio",
                            "options": ["Too long", "Reasonable", "Short"]
                        }
                    ]
                },
                {
                    "section": "Feedback",
                    "fields": [
                        {
                            "name": "comments",
                            "label": "Additional Comments",
                            "type": "textarea",
                            "max_length": 500
                        }
                    ]
                }
            ],
            "languages": ["en", "es", "fr", "de", "it", "pt", "hi", "ar", "zh", "ja"]
        }
    
    @staticmethod
    def get_food_feedback_form() -> dict:
        """Get food quality feedback form"""
        return {
            "form_id": "food_feedback",
            "title": "Food Service Feedback",
            "sections": [
                {
                    "section": "Food Quality",
                    "fields": [
                        {
                            "name": "overall_rating",
                            "label": "Overall Food Quality",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "taste",
                            "label": "Taste",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "hygiene",
                            "label": "Hygiene & Cleanliness",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "temperature",
                            "label": "Food Temperature",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "variety",
                            "label": "Menu Variety",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "portion_size",
                            "label": "Portion Size",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        }
                    ]
                },
                {
                    "section": "Comments",
                    "fields": [
                        {
                            "name": "comments",
                            "label": "What could be improved?",
                            "type": "textarea",
                            "max_length": 500
                        }
                    ]
                }
            ],
            "languages": ["en", "es", "fr", "de", "it", "pt", "hi", "ar", "zh", "ja"]
        }
    
    @staticmethod
    def get_room_feedback_form() -> dict:
        """Get room quality feedback form"""
        return {
            "form_id": "room_feedback",
            "title": "Room Quality Feedback",
            "sections": [
                {
                    "section": "Room Conditions",
                    "fields": [
                        {
                            "name": "room_id",
                            "label": "Room Number",
                            "type": "text",
                            "readonly": True
                        },
                        {
                            "name": "cleanliness",
                            "label": "Overall Cleanliness",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "furniture_condition",
                            "label": "Furniture Condition",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "bathroom_condition",
                            "label": "Bathroom Condition",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "bed_comfort",
                            "label": "Bed Comfort",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        },
                        {
                            "name": "lighting",
                            "label": "Lighting Quality",
                            "type": "rating",
                            "min": 1,
                            "max": 5
                        }
                    ]
                },
                {
                    "section": "Issues",
                    "fields": [
                        {
                            "name": "issues",
                            "label": "Any maintenance issues?",
                            "type": "checkbox",
                            "options": ["Dirty", "Broken furniture", "Plumbing issues", 
                                      "Lighting issues", "Temperature control", "Noise issues"]
                        },
                        {
                            "name": "comments",
                            "label": "Additional Details",
                            "type": "textarea",
                            "max_length": 500
                        }
                    ]
                }
            ],
            "languages": ["en", "es", "fr", "de", "it", "pt", "hi", "ar", "zh", "ja"]
        }
    
    @staticmethod
    def get_complaint_form() -> dict:
        """Get complaint submission form"""
        return {
            "form_id": "complaint_form",
            "title": "Submit a Complaint",
            "sections": [
                {
                    "section": "Complaint Details",
                    "fields": [
                        {
                            "name": "complaint_type",
                            "label": "Type of Complaint",
                            "type": "select",
                            "options": ["Medical Staff", "Nursing Staff", "Cleanliness", 
                                      "Food Service", "Equipment", "Other"]
                        },
                        {
                            "name": "target_person",
                            "label": "Staff Member (if applicable)",
                            "type": "text"
                        },
                        {
                            "name": "incident_date",
                            "label": "Date of Incident",
                            "type": "date"
                        },
                        {
                            "name": "severity",
                            "label": "Severity Level",
                            "type": "radio",
                            "options": ["Low", "Medium", "High", "Critical"]
                        }
                    ]
                },
                {
                    "section": "Description",
                    "fields": [
                        {
                            "name": "description",
                            "label": "Detailed Description",
                            "type": "textarea",
                            "max_length": 2000,
                            "placeholder": "Please describe the incident in detail"
                        },
                        {
                            "name": "witnesses",
                            "label": "Were there any witnesses?",
                            "type": "text",
                            "placeholder": "Names and roles of witnesses (optional)"
                        }
                    ]
                },
                {
                    "section": "Contact Information",
                    "fields": [
                        {
                            "name": "name",
                            "label": "Your Name",
                            "type": "text"
                        },
                        {
                            "name": "contact",
                            "label": "Contact Number/Email",
                            "type": "text"
                        },
                        {
                            "name": "anonymous",
                            "label": "File anonymously",
                            "type": "checkbox"
                        }
                    ]
                }
            ],
            "languages": ["en", "es", "fr", "de", "it", "pt", "hi", "ar", "zh", "ja"]
        }


# Initialize global QR code generator
qr_generator = QRCodeGenerator()

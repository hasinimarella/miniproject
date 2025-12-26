from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
from flask_cors import CORS
from datetime import datetime
import threading
import time
import tempfile
import sys
import os
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(backend_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, backend_dir)

# Paths for persisted datasets
FEEDBACK_LOG_PATH = os.path.join(project_dir, 'data', 'patient_feedback_log.jsonl')

# Import configuration
try:
    from config.config import config
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = {'development': {}}

# Import all modules with error handling
try:
    from sentiment_analyzer import sentiment_analyzer
    logger.info("Sentiment analyzer loaded")
except Exception as e:
    logger.error(f"Failed to load sentiment_analyzer: {e}")
    sentiment_analyzer = None

try:
    from doctor_analyzer import doctor_analyzer
    logger.info("Doctor analyzer loaded")
except Exception as e:
    logger.error(f"Failed to load doctor_analyzer: {e}")
    doctor_analyzer = None

try:
    from facility_monitor import facility_monitor
    logger.info("Facility monitor loaded")
except Exception as e:
    logger.error(f"Failed to load facility_monitor: {e}")
    facility_monitor = None

try:
    from dashboard import trend_analyzer, OperationsDashboard
    logger.info("Dashboard modules loaded")
except Exception as e:
    logger.error(f"Failed to load dashboard: {e}")
    trend_analyzer = None
    OperationsDashboard = None

try:
    from alert_system import alert_manager, AlertType, AlertSeverity
    logger.info("Alert system loaded")
except Exception as e:
    logger.error(f"Failed to load alert_system: {e}")
    alert_manager = None
    AlertType = None
    AlertSeverity = None

try:
    from qr_system import qr_generator, FeedbackForm
    logger.info("QR system loaded")
except Exception as e:
    logger.error(f"Failed to load qr_system: {e}")
    qr_generator = None
    FeedbackForm = None

# Initialize Flask app
app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

# Load configuration
try:
    app.config.from_object(config.get('development', {}))
    logger.info("Flask configuration applied")
except Exception as e:
    logger.warning(f"Flask configuration warning: {e}")

# Initialize dashboard if modules are available
operations_dashboard = None
if trend_analyzer and OperationsDashboard and doctor_analyzer and facility_monitor:
    try:
        operations_dashboard = OperationsDashboard(trend_analyzer, doctor_analyzer, facility_monitor)
        logger.info("Operations dashboard initialized")
    except Exception as e:
        logger.error(f"Failed to initialize operations dashboard: {e}")


# ==================== HEALTH CHECK ====================
@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Hospital Sentiment Analysis System'
    }), 200


# ==================== REVIEW SUBMISSION ====================
@app.route('/api/reviews/submit', methods=['POST'])
def submit_review():
    """Submit a new review"""
    try:
        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        data = request.json
        review_text = data.get('review_text', '').strip()
        category = data.get('category', 'general').strip()
        patient_id = data.get('patient_id')
        doctor_id = data.get('doctor_id')
        
        # Validate review text
        if not review_text:
            return jsonify({'error': 'Review text is required and cannot be empty'}), 400
        
        if len(review_text) > 5000:
            return jsonify({'error': 'Review text exceeds maximum length of 5000 characters'}), 400
        
        # Attempt sentiment analysis; if unavailable, continue with a fallback analysis
        analysis = None
        if sentiment_analyzer:
            try:
                analysis = sentiment_analyzer.comprehensive_analysis(review_text)
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                analysis = None

        if not analysis:
            # Provide a safe fallback so submissions still persist when NLP service is down
            analysis = {
                'original_language': 'en',
                'translated_text': review_text,
                'overall_score': 0.0,
                'sentiment_label': 'UNKNOWN',
                'confidence': 0.0,
                'vader': {},
                'textblob': {},
                'transformer': {},
                'emotions': {},
                'dominant_emotion': None,
                'keywords': [],
                'subjectivity': 0.0,
                'analysis_unavailable': True
            }
        
        # Add to trend analyzer if available
        if trend_analyzer:
            try:
                review_id = f"REV_{int(datetime.now().timestamp() * 1000)}"
                # Capture contextual metadata for richer analytics / admin views
                metadata = {
                    'input_method': data.get('input_method', 'text'),
                    'patient_id': patient_id,
                    'patient_name': data.get('patient_name'),
                    'rating': data.get('rating'),
                    'language': analysis.get('original_language')
                }
                trend_analyzer.add_review_analysis(review_id, analysis, category, metadata)
            except Exception as e:
                logger.warning(f"Could not add review to trends: {e}")

        # Persist full feedback + analysis to training log (for future model improvements)
        try:
            log_entry = {
                'review_id': f"REV_{int(datetime.now().timestamp() * 1000)}",
                'created_at': datetime.now().isoformat(),
                'review_text': review_text,
                'category': category,
                'patient_id': patient_id,
                'patient_name': data.get('patient_name'),
                'doctor_id': doctor_id,
                'rating': data.get('rating'),
                'input_method': data.get('input_method', 'text'),
                'language_detected': analysis.get('original_language'),
                'translated_text': analysis.get('translated_text'),
                'overall_score': analysis.get('overall_score'),
                'sentiment_label': analysis.get('sentiment_label'),
                'emotions': analysis.get('emotions'),
                'keywords': analysis.get('keywords'),
                'subjectivity': analysis.get('subjectivity'),
                'model_version': 'v2.0-multilingual'
            }

            os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
            with open(FEEDBACK_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to persist feedback log: {e}")
        
        # Check for critical sentiment if alert_manager is available
        alert = None
        if alert_manager:
            try:
                alert = alert_manager.check_sentiment_alert(
                    analysis['overall_score'],
                    f"REV_{datetime.now().timestamp()}"
                )
            except Exception as e:
                logger.warning(f"Could not check sentiment alert: {e}")
        
        # Track doctor sentiment if applicable
        if doctor_id and doctor_analyzer:
            try:
                doctor_analyzer.track_sentiment(doctor_id, analysis['overall_score'])
            except Exception as e:
                logger.warning(f"Could not track doctor sentiment: {e}")
        
        return jsonify({
            'status': 'success',
            'review_id': f"REV_{int(datetime.now().timestamp() * 1000)}",
            'sentiment_analysis': analysis,
            'alert': alert.to_dict() if alert else None,
            'timestamp': datetime.now().isoformat()
        }), 201
    
    except Exception as e:
        logger.error(f"Error in submit_review: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# ==================== SENTIMENT ANALYTICS ====================
@app.route('/api/analytics/sentiment', methods=['GET'])
def get_sentiment_analytics():
    """Get sentiment analytics"""
    try:
        if not trend_analyzer:
            return jsonify({'error': 'Analytics service is unavailable'}), 503
        
        days = request.args.get('days', 30, type=int)
        category = request.args.get('category')
        
        # Validate days parameter
        if days < 1 or days > 365:
            return jsonify({'error': 'Days parameter must be between 1 and 365'}), 400
        
        try:
            trends = trend_analyzer.get_sentiment_trends(days, category)
            emotions = trend_analyzer.get_emotion_analysis(category, days)
        except Exception as e:
            logger.error(f"Error getting sentiment trends: {e}")
            return jsonify({'error': 'Failed to retrieve sentiment data'}), 500
        
        return jsonify({
            'status': 'success',
            'sentiment_trends': trends,
            'emotion_analysis': emotions,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_sentiment_analytics: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== ISSUE CLUSTERING ====================
@app.route('/api/analytics/issue-clusters', methods=['GET'])
def get_issue_clusters():
    """Get clustered issues"""
    try:
        if not trend_analyzer:
            return jsonify({'error': 'Clustering service is unavailable'}), 503
        
        max_clusters = request.args.get('max_clusters', 5, type=int)
        
        # Validate max_clusters
        if max_clusters < 1 or max_clusters > 20:
            return jsonify({'error': 'max_clusters must be between 1 and 20'}), 400
        
        try:
            clusters = trend_analyzer.cluster_issues(max_clusters)
        except Exception as e:
            logger.error(f"Error clustering issues: {e}")
            return jsonify({'error': 'Failed to cluster issues'}), 500
        
        return jsonify({
            'status': 'success',
            'issue_clusters': clusters,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_issue_clusters: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== DOCTOR ANALYTICS ====================
@app.route('/api/doctors/<doctor_id>/performance', methods=['GET'])
def get_doctor_performance(doctor_id):
    """Get doctor performance dashboard"""
    try:
        if not doctor_analyzer:
            return jsonify({'error': 'Doctor analytics service is unavailable'}), 503
        
        if not doctor_id or len(doctor_id.strip()) == 0:
            return jsonify({'error': 'Doctor ID is required'}), 400
        
        try:
            performance = doctor_analyzer.get_doctor_performance_dashboard(doctor_id)
        except Exception as e:
            logger.error(f"Error getting doctor performance: {e}")
            return jsonify({'error': 'Failed to retrieve doctor performance data'}), 500
        
        return jsonify({
            'status': 'success',
            'performance_dashboard': performance,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_doctor_performance: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/doctors/<doctor_id>/duty-shift', methods=['POST'])
def log_duty_shift(doctor_id):
    """Log doctor duty shift"""
    try:
        if not doctor_analyzer:
            return jsonify({'error': 'Doctor service is unavailable'}), 503
        
        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        if not doctor_id or len(doctor_id.strip()) == 0:
            return jsonify({'error': 'Doctor ID is required'}), 400
        
        data = request.json
        shift_date = data.get('shift_date')
        hours = data.get('hours')
        patient_count = data.get('patient_count')
        
        if not shift_date or hours is None or patient_count is None:
            return jsonify({'error': 'shift_date, hours, and patient_count are required'}), 400
        
        try:
            result = doctor_analyzer.register_duty_shift(
                doctor_id,
                shift_date,
                hours,
                patient_count,
                data.get('emergency_cases', 0)
            )
            
            # Check for burnout alert
            burnout = doctor_analyzer.calculate_burnout_risk(doctor_id)
            alert = None
            if alert_manager:
                try:
                    alert = alert_manager.check_burnout_alert(
                        doctor_id,
                        burnout['risk_score'],
                        burnout['risk_level']
                    )
                except Exception as e:
                    logger.warning(f"Could not check burnout alert: {e}")
            
            return jsonify({
                'status': 'success',
                'shift_logged': result,
                'burnout_analysis': burnout,
                'alert': alert.to_dict() if alert else None,
                'timestamp': datetime.now().isoformat()
            }), 201
        except Exception as e:
            logger.error(f"Error logging duty shift: {e}")
            return jsonify({'error': 'Failed to log duty shift'}), 500
    
    except Exception as e:
        logger.error(f"Error in log_duty_shift: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/complaints/doctor', methods=['POST'])
def file_doctor_complaint():
    """File complaint against doctor"""
    try:
        if not doctor_analyzer:
            return jsonify({'error': 'Complaint service is unavailable'}), 503
        
        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        data = request.json
        doctor_id = data.get('doctor_id')
        complaint_type = data.get('complaint_type')
        description = data.get('description')
        severity = data.get('severity', 'MEDIUM')
        
        if not doctor_id or not complaint_type or not description:
            return jsonify({'error': 'doctor_id, complaint_type, and description are required'}), 400
        
        try:
            result = doctor_analyzer.file_complaint(
                doctor_id,
                complaint_type,
                description,
                severity
            )
            
            # Create alert for complaint
            alert = None
            if alert_manager:
                try:
                    alert = alert_manager.check_complaint_alert(
                        doctor_id,
                        complaint_type,
                        severity
                    )
                except Exception as e:
                    logger.warning(f"Could not create complaint alert: {e}")
            
            return jsonify({
                'status': 'success',
                'complaint_filed': result,
                'alert': alert.to_dict() if alert else None,
                'timestamp': datetime.now().isoformat()
            }), 201
        except Exception as e:
            logger.error(f"Error filing complaint: {e}")
            return jsonify({'error': 'Failed to file complaint'}), 500
    
    except Exception as e:
        logger.error(f"Error in file_doctor_complaint: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== FACILITY QUALITY ====================
@app.route('/api/facility/food-quality', methods=['POST'])
def submit_food_quality():
    """Submit food quality review"""
    try:
        if not facility_monitor:
            return jsonify({'error': 'Facility service is unavailable'}), 503
        
        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        data = request.json
        quality_score = data.get('quality_score')
        aspects = data.get('aspects', {})
        comments = data.get('comments', '').strip()
        
        if quality_score is None:
            return jsonify({'error': 'quality_score is required'}), 400
        
        if not (1 <= quality_score <= 5):
            return jsonify({'error': 'quality_score must be between 1 and 5'}), 400
        
        if len(comments) > 1000:
            return jsonify({'error': 'Comments exceed maximum length of 1000 characters'}), 400
        
        try:
            result = facility_monitor.submit_food_quality_review(
                f"FOOD_{int(datetime.now().timestamp() * 1000)}",
                quality_score,
                aspects,
                comments
            )
            
            return jsonify({
                'status': 'success',
                'review_submitted': result,
                'timestamp': datetime.now().isoformat()
            }), 201
        except Exception as e:
            logger.error(f"Error submitting food quality review: {e}")
            return jsonify({'error': 'Failed to submit food quality review'}), 500
    
    except Exception as e:
        logger.error(f"Error in submit_food_quality: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/facility/room-quality', methods=['POST'])
def submit_room_quality():
    """Submit room quality review"""
    try:
        if not facility_monitor:
            return jsonify({'error': 'Facility service is unavailable'}), 503
        
        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        data = request.json
        room_id = data.get('room_id')
        cleanliness_score = data.get('cleanliness_score')
        aspects = data.get('aspects', {})
        comments = data.get('comments', '').strip()
        
        if not room_id or cleanliness_score is None:
            return jsonify({'error': 'room_id and cleanliness_score are required'}), 400
        
        if not (1 <= cleanliness_score <= 5):
            return jsonify({'error': 'cleanliness_score must be between 1 and 5'}), 400
        
        try:
            result = facility_monitor.submit_room_quality_review(
                f"ROOM_{int(datetime.now().timestamp() * 1000)}",
                room_id,
                cleanliness_score,
                aspects,
                comments
            )
            
            return jsonify({
                'status': 'success',
                'review_submitted': result,
                'timestamp': datetime.now().isoformat()
            }), 201
        except Exception as e:
            logger.error(f"Error submitting room quality review: {e}")
            return jsonify({'error': 'Failed to submit room quality review'}), 500
    
    except Exception as e:
        logger.error(f"Error in submit_room_quality: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/facility/food-analytics', methods=['GET'])
def get_food_analytics():
    """Get food quality analytics"""
    try:
        if not facility_monitor:
            return jsonify({'error': 'Analytics service is unavailable'}), 503
        
        days = request.args.get('days', 30, type=int)
        
        if days < 1 or days > 365:
            return jsonify({'error': 'Days parameter must be between 1 and 365'}), 400
        
        try:
            analysis = facility_monitor.analyze_food_quality_trends(days)
            recommendations = facility_monitor.get_food_quality_recommendations(days)
        except Exception as e:
            logger.error(f"Error getting food analytics: {e}")
            return jsonify({'error': 'Failed to retrieve food analytics'}), 500
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_food_analytics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/facility/room-analytics', methods=['GET'])
def get_room_analytics():
    """Get room quality analytics"""
    try:
        if not facility_monitor:
            return jsonify({'error': 'Analytics service is unavailable'}), 503
        
        days = request.args.get('days', 30, type=int)
        
        if days < 1 or days > 365:
            return jsonify({'error': 'Days parameter must be between 1 and 365'}), 400
        
        try:
            analysis = facility_monitor.analyze_room_quality_trends(days)
            recommendations = facility_monitor.get_room_quality_recommendations(days)
        except Exception as e:
            logger.error(f"Error getting room analytics: {e}")
            return jsonify({'error': 'Failed to retrieve room analytics'}), 500
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_room_analytics: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== ALERT SYSTEM ====================
@app.route('/api/alerts/active', methods=['GET'])
def get_active_alerts():
    """Get active alerts"""
    try:
        if not alert_manager:
            return jsonify({'error': 'Alert service is unavailable'}), 503
        
        severity_param = request.args.get('severity')
        alerts = []
        
        try:
            if severity_param and AlertSeverity:
                alerts = alert_manager.get_active_alerts(AlertSeverity[severity_param.upper()])
            else:
                alerts = alert_manager.get_active_alerts()
        except (KeyError, ValueError):
            return jsonify({'error': f'Invalid severity level: {severity_param}'}), 400
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return jsonify({'error': 'Failed to retrieve alerts'}), 500
        
        return jsonify({
            'status': 'success',
            'alerts': [a.to_dict() if hasattr(a, 'to_dict') else a for a in alerts],
            'total_count': len(alerts),
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_active_alerts: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        if not alert_manager:
            return jsonify({'error': 'Alert service is unavailable'}), 503
        
        if not alert_id or len(alert_id.strip()) == 0:
            return jsonify({'error': 'Alert ID is required'}), 400
        
        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        data = request.json
        user_id = data.get('user_id', 'system')
        
        try:
            success = alert_manager.acknowledge_alert(alert_id, user_id)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': f'Alert {alert_id} acknowledged',
                    'timestamp': datetime.now().isoformat()
                }), 200
            else:
                return jsonify({'error': 'Alert not found'}), 404
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return jsonify({'error': 'Failed to acknowledge alert'}), 500
    
    except Exception as e:
        logger.error(f"Error in acknowledge_alert: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """Get alert statistics"""
    try:
        if not alert_manager:
            return jsonify({'error': 'Alert service is unavailable'}), 503
        
        hours = request.args.get('hours', 24, type=int)
        
        if hours < 1 or hours > 8760:  # Max 1 year
            return jsonify({'error': 'Hours must be between 1 and 8760'}), 400
        
        try:
            stats = alert_manager.get_alert_statistics(hours)
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return jsonify({'error': 'Failed to retrieve alert statistics'}), 500
        
        return jsonify({
            'status': 'success',
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_alert_statistics: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== DASHBOARD ====================
@app.route('/api/dashboard/overview', methods=['GET'])
def get_dashboard_overview():
    """Get comprehensive dashboard overview"""
    try:
        if not operations_dashboard:
            return jsonify({'error': 'Dashboard service is unavailable'}), 503
        
        try:
            overview = operations_dashboard.get_dashboard_overview()
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return jsonify({'error': 'Failed to retrieve dashboard data'}), 500
        
        return jsonify({
            'status': 'success',
            'dashboard': overview,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_dashboard_overview: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== ADMIN – RECENT REVIEWS ====================
@app.route('/api/admin/reviews/recent', methods=['GET'])
def get_recent_reviews():
    """Get recent patient reviews with sentiment and input method for admin view"""
    try:
        if not trend_analyzer:
            return jsonify({'error': 'Analytics service is unavailable'}), 503

        # Limit number of records returned
        limit = request.args.get('limit', 20, type=int)
        if limit < 1 or limit > 200:
            return jsonify({'error': 'limit must be between 1 and 200'}), 400

        # Get all reviews from in‑memory trend analyzer and sort by timestamp (newest first)
        try:
            all_reviews = getattr(trend_analyzer, 'reviews', [])
        except Exception as e:
            logger.error(f"Error accessing trend reviews: {e}")
            return jsonify({'error': 'Failed to access review analytics'}), 500

        sorted_reviews = sorted(
            all_reviews,
            key=lambda r: r.get('timestamp', ''),
            reverse=True
        )

        recent = sorted_reviews[:limit]

        # Project only the fields needed by admin UI
        projected = []
        for r in recent:
            projected.append({
                'review_id': r.get('review_id'),
                'timestamp': r.get('timestamp'),
                'category': r.get('category'),
                'sentiment_label': r.get('sentiment_label'),
                'sentiment_score': r.get('sentiment_score'),
                'input_method': r.get('input_method', 'text'),
                'rating': r.get('rating'),
                'patient_name': r.get('patient_name') or 'Anonymous',
                'language': r.get('language') or r.get('original_language')
            })

        return jsonify({
            'status': 'success',
            'total_reviews_tracked': len(all_reviews),
            'count': len(projected),
            'reviews': projected,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in get_recent_reviews: {e}")
        return jsonify({'error': 'Failed to retrieve recent reviews'}), 500


# ==================== QR CODE & FEEDBACK ====================
@app.route('/api/qr/generate', methods=['POST'])
def generate_qr():
    """Generate QR code for feedback"""
    try:
        if not qr_generator:
            return jsonify({'error': 'QR generation service is unavailable'}), 503
        
        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        data = request.json
        qr_type = data.get('type', 'general').strip()
        metadata = data.get('metadata', {})
        
        if not qr_type:
            return jsonify({'error': 'type parameter is required'}), 400
        
        try:
            if qr_type == 'doctor':
                qr_code, url = qr_generator.generate_doctor_feedback_qr(
                    metadata.get('doctor_id'),
                    metadata.get('doctor_name')
                )
            elif qr_type == 'room':
                qr_code, url = qr_generator.generate_room_feedback_qr(
                    metadata.get('room_id')
                )
            elif qr_type == 'food':
                qr_code, url = qr_generator.generate_food_feedback_qr()
            elif qr_type == 'complaint':
                qr_code, url = qr_generator.generate_complaint_qr()
            else:
                qr_code, url = qr_generator.generate_feedback_qr(qr_type, metadata)
            
            return jsonify({
                'status': 'success',
                'qr_code': qr_code,
                'feedback_url': url,
                'type': qr_type,
                'timestamp': datetime.now().isoformat()
            }), 200
        except Exception as e:
            logger.error(f"Error generating QR code: {e}")
            return jsonify({'error': 'Failed to generate QR code'}), 500
    
    except Exception as e:
        logger.error(f"Error in generate_qr: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forms/<form_type>', methods=['GET'])
def get_feedback_form(form_type):
    """Get feedback form structure"""
    try:
        if not FeedbackForm:
            return jsonify({'error': 'Form service is unavailable'}), 503
        
        if not form_type or len(form_type.strip()) == 0:
            return jsonify({'error': 'form_type is required'}), 400
        
        forms = {
            'general': FeedbackForm.get_general_feedback_form,
            'doctor': FeedbackForm.get_doctor_feedback_form,
            'food': FeedbackForm.get_food_feedback_form,
            'room': FeedbackForm.get_room_feedback_form,
            'complaint': FeedbackForm.get_complaint_form
        }
        
        if form_type not in forms:
            return jsonify({'error': f'Form type not found: {form_type}'}), 404
        
        try:
            form = forms[form_type]()
        except Exception as e:
            logger.error(f"Error getting feedback form: {e}")
            return jsonify({'error': 'Failed to retrieve form structure'}), 500
        
        return jsonify({
            'status': 'success',
            'form': form,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_feedback_form: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== AUTHENTICATION & SESSION ====================
@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user and clear session"""
    try:
        logger.info("User logout requested")
        return jsonify({
            'status': 'success',
            'message': 'Successfully logged out',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== SYSTEM STATUS ====================
@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get system status and health information"""
    try:
        status = {
            'status': 'operational',
            'services': {
                'sentiment_analyzer': sentiment_analyzer is not None,
                'doctor_analyzer': doctor_analyzer is not None,
                'facility_monitor': facility_monitor is not None,
                'dashboard': operations_dashboard is not None,
                'alert_manager': alert_manager is not None,
                'qr_generator': qr_generator is not None
            },
            'timestamp': datetime.now().isoformat(),
            'version': '2.0-optimized'
        }
        
        # Check if all critical services are available
        critical_services = [sentiment_analyzer, doctor_analyzer, facility_monitor]
        if not all(critical_services):
            status['status'] = 'degraded'
        
        return jsonify(status), 200
    
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return jsonify({'error': 'Failed to retrieve system status'}), 500


# ==================== FRONTEND ROUTES ====================
@app.route('/', methods=['GET'])
def index():
    """Landing page with doctor login and patient feedback options"""
    try:
        logger.info("Loading landing page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading landing page: {e}")
        return f"Error loading landing page: {str(e)}", 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Doctor dashboard with analytics"""
    try:
        logger.info("Loading doctor dashboard")
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return f"Error loading dashboard: {str(e)}", 500


@app.route('/feedback/<feedback_type>', methods=['GET'])
def feedback_page(feedback_type):
    """Feedback submission page"""
    try:
        if not feedback_type or len(feedback_type.strip()) == 0:
            return "Invalid feedback type", 400
        
        logger.info(f"Loading feedback page for type: {feedback_type}")
        return render_template('feedback.html', feedback_type=feedback_type)
    except Exception as e:
        logger.error(f"Error loading feedback page: {e}")
        return f"Error loading feedback page: {str(e)}", 500


@app.route('/admin', methods=['GET'])
def admin_panel():
    """Admin dashboard page"""
    try:
        logger.info("Loading admin panel")
        return render_template('admin.html')
    except Exception as e:
        logger.error(f"Error loading admin panel: {e}")
        return f"Error loading admin panel: {str(e)}", 500


# Note: static PWA assets (manifest, service-worker, icons) are served from the
# `frontend/static` folder so they are available under `/static/...`.


# ==================== ERROR HANDLERS ====================
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request"""
    logger.warning(f"Bad request: {error}")
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found"""
    logger.warning(f"Resource not found: {error}")
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server Error"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error', 'message': 'An unexpected error occurred'}), 500


@app.errorhandler(503)
def service_unavailable(error):
    """Handle 503 Service Unavailable"""
    logger.error(f"Service unavailable: {error}")
    return jsonify({'error': 'Service temporarily unavailable'}), 503


@app.route('/api/status', methods=['GET'])
def api_status():
    """Return service availability for frontend checks"""
    try:
        status = {
            'sentiment_analyzer': bool(sentiment_analyzer),
            'doctor_analyzer': bool(doctor_analyzer),
            'facility_monitor': bool(facility_monitor),
            'alert_manager': bool(alert_manager),
            'qr_generator': bool(qr_generator),
            'dashboard': bool(operations_dashboard)
        }
        return jsonify({'status': 'success', 'services': status, 'timestamp': datetime.now().isoformat()}), 200
    except Exception as e:
        logger.error(f"Error in api_status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def reprocess_pending_feedback_once():
    """Read the feedback log and re-run sentiment analysis for entries
    that were saved while the analyzer was unavailable. Returns number
    of records updated."""
    if not sentiment_analyzer:
        logger.info("Sentiment analyzer not available; skipping reprocess")
        return 0

    if not os.path.exists(FEEDBACK_LOG_PATH):
        logger.info("Feedback log does not exist; nothing to reprocess")
        return 0

    updated_count = 0
    try:
        # Read all entries
        with open(FEEDBACK_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        records = [json.loads(l) for l in lines]

        for rec in records:
            # Detect fallback/unprocessed entries
            if rec.get('analysis_unavailable') or rec.get('sentiment_label') in (None, 'UNKNOWN'):
                try:
                    text = rec.get('review_text', '')
                    analysis = sentiment_analyzer.comprehensive_analysis(text)
                    # Update record fields with fresh analysis
                    rec['language_detected'] = analysis.get('original_language')
                    rec['translated_text'] = analysis.get('translated_text')
                    rec['overall_score'] = analysis.get('overall_score')
                    rec['sentiment_label'] = analysis.get('sentiment_label')
                    rec['emotions'] = analysis.get('emotions')
                    rec['keywords'] = analysis.get('keywords')
                    rec['subjectivity'] = analysis.get('subjectivity')
                    rec['model_version'] = rec.get('model_version', 'v2.0-multilingual')
                    rec['analysis_unavailable'] = False
                    rec['reprocessed_at'] = datetime.now().isoformat()
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to reprocess review {rec.get('review_id')}: {e}")
                    continue

        if updated_count > 0:
            # Write back all records safely to a temp file then replace
            dirpath = os.path.dirname(FEEDBACK_LOG_PATH)
            os.makedirs(dirpath, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=dirpath, text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as tf:
                    for r in records:
                        tf.write(json.dumps(r, ensure_ascii=False) + '\n')
                os.replace(tmp_path, FEEDBACK_LOG_PATH)
                logger.info(f"Reprocessed {updated_count} feedback entries and updated log")
            except Exception as e:
                logger.error(f"Failed to write updated feedback log: {e}")
                # Clean up temp file if something went wrong
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Error during reprocessing pending feedback: {e}")

    return updated_count


@app.route('/api/admin/reprocess-pending', methods=['POST'])
def api_reprocess_pending():
    """Admin endpoint to trigger immediate reprocessing of pending feedback."""
    try:
        count = reprocess_pending_feedback_once()
        return jsonify({'status': 'success', 'updated': count, 'timestamp': datetime.now().isoformat()}), 200
    except Exception as e:
        logger.error(f"Error in api_reprocess_pending: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _reprocess_background_loop(poll_interval=300):
    """Background loop that periodically attempts to reprocess pending feedback."""
    logger.info("Background reprocess loop started")
    while True:
        try:
            if sentiment_analyzer:
                reprocess_pending_feedback_once()
                time.sleep(poll_interval)
            else:
                # If analyzer still not ready, wait shorter and retry
                time.sleep(60)
        except Exception as e:
            logger.error(f"Error in reprocess background loop: {e}")
            time.sleep(60)


def _start_reprocess_thread():
    # Start background thread as daemon so it doesn't block shutdown
    try:
        t = threading.Thread(target=_reprocess_background_loop, daemon=True)
        t.start()
        logger.info("Reprocess background thread started")
    except Exception as e:
        logger.error(f"Failed to start reprocess background thread: {e}")


# ==================== MAIN ====================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Hospital Sentiment Analysis System - Starting Up")
    logger.info("=" * 60)
    
    # Print startup information
    print("\n" + "=" * 60)
    print("Hospital Sentiment Analysis System v2.0")
    print("=" * 60)
    print(f"Starting server at http://localhost:5000")
    print(f"Configuration: {os.getenv('FLASK_ENV', 'development')}")
    print("=" * 60)
    
    # Check service availability
    print("\nService Status:")
    print(f"✓ Sentiment Analyzer: {'Available' if sentiment_analyzer else 'UNAVAILABLE'}")
    print(f"✓ Doctor Analyzer: {'Available' if doctor_analyzer else 'UNAVAILABLE'}")
    print(f"✓ Facility Monitor: {'Available' if facility_monitor else 'UNAVAILABLE'}")
    print(f"✓ Alert Manager: {'Available' if alert_manager else 'UNAVAILABLE'}")
    print(f"✓ QR Generator: {'Available' if qr_generator else 'UNAVAILABLE'}")
    print(f"✓ Dashboard: {'Available' if operations_dashboard else 'UNAVAILABLE'}")
    print("=" * 60 + "\n")
    
    logger.info("All services initialized successfully")
    
    # Start background reprocess thread and run Flask app with error handling
    try:
        _start_reprocess_thread()
    except Exception as e:
        logger.warning(f"Could not start reprocess thread at startup: {e}")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Failed to start Flask server: {e}")
        print(f"\nERROR: Failed to start server - {e}")
        exit(1)

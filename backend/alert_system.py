from typing import Dict, List, Callable
from datetime import datetime
from enum import Enum
import json


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertType(Enum):
    """Types of alerts"""
    SENTIMENT_CRITICAL = "SENTIMENT_CRITICAL"
    DOCTOR_BURNOUT = "DOCTOR_BURNOUT"
    COMPLAINT_FILED = "COMPLAINT_FILED"
    FOOD_QUALITY = "FOOD_QUALITY"
    ROOM_QUALITY = "ROOM_QUALITY"
    ISSUE_CLUSTER = "ISSUE_CLUSTER"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class Alert:
    """Alert object"""
    
    def __init__(self, alert_type: AlertType, severity: AlertSeverity, 
                 message: str, details: Dict = None):
        self.alert_id = f"ALT_{datetime.now().timestamp()}"
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.created_at = datetime.now().isoformat()
        self.acknowledged = False
        self.acknowledged_at = None
        self.acknowledged_by = None
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'created_at': self.created_at,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at,
            'acknowledged_by': self.acknowledged_by
        }
    
    def acknowledge(self, user_id: str) -> None:
        """Acknowledge the alert"""
        self.acknowledged = True
        self.acknowledged_at = datetime.now().isoformat()
        self.acknowledged_by = user_id


class AlertManager:
    """
    Manages critical alerts and notifications
    """
    
    def __init__(self):
        self.alerts = []
        self.alert_handlers = {}
        self.threshold_config = {
            'sentiment_critical': -0.7,
            'sentiment_warning': -0.4,
            'burnout_critical': 0.7,
            'burnout_high': 0.5,
            'complaint_critical': 1,
            'food_quality_critical': 2.0,
            'room_quality_critical': 2.0,
            'issue_cluster_critical': 5
        }
    
    def register_handler(self, alert_type: AlertType, handler: Callable) -> None:
        """
        Register a handler for specific alert type
        Handler should accept Alert object as parameter
        """
        if alert_type not in self.alert_handlers:
            self.alert_handlers[alert_type] = []
        self.alert_handlers[alert_type].append(handler)
    
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                    message: str, details: Dict = None) -> Alert:
        """
        Create and register an alert
        """
        alert = Alert(alert_type, severity, message, details)
        self.alerts.append(alert)
        
        # Trigger registered handlers
        if alert_type in self.alert_handlers:
            for handler in self.alert_handlers[alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Error executing alert handler: {e}")
        
        return alert
    
    def check_sentiment_alert(self, sentiment_score: float, review_id: str = None) -> Alert:
        """
        Check if sentiment score warrants an alert
        """
        if sentiment_score < self.threshold_config['sentiment_critical']:
            return self.create_alert(
                AlertType.SENTIMENT_CRITICAL,
                AlertSeverity.HIGH,
                f"Critical negative sentiment detected (Score: {sentiment_score})",
                {'sentiment_score': sentiment_score, 'review_id': review_id}
            )
        return None
    
    def check_burnout_alert(self, doctor_id: str, burnout_score: float, 
                           burnout_level: str) -> Alert:
        """
        Check if burnout score warrants an alert
        """
        if burnout_level == 'CRITICAL':
            return self.create_alert(
                AlertType.DOCTOR_BURNOUT,
                AlertSeverity.CRITICAL,
                f"CRITICAL: Doctor {doctor_id} showing severe burnout symptoms",
                {'doctor_id': doctor_id, 'burnout_score': burnout_score, 'risk_level': burnout_level}
            )
        elif burnout_level == 'HIGH':
            return self.create_alert(
                AlertType.DOCTOR_BURNOUT,
                AlertSeverity.HIGH,
                f"Doctor {doctor_id} showing high burnout risk",
                {'doctor_id': doctor_id, 'burnout_score': burnout_score, 'risk_level': burnout_level}
            )
        return None
    
    def check_complaint_alert(self, doctor_id: str, complaint_type: str,
                             severity: str) -> Alert:
        """
        Check if complaint warrants an alert
        """
        if severity == 'CRITICAL':
            return self.create_alert(
                AlertType.COMPLAINT_FILED,
                AlertSeverity.CRITICAL,
                f"CRITICAL complaint filed against Doctor {doctor_id}: {complaint_type}",
                {'doctor_id': doctor_id, 'complaint_type': complaint_type, 'severity': severity}
            )
        elif severity == 'HIGH':
            return self.create_alert(
                AlertType.COMPLAINT_FILED,
                AlertSeverity.HIGH,
                f"Serious complaint filed against Doctor {doctor_id}",
                {'doctor_id': doctor_id, 'complaint_type': complaint_type, 'severity': severity}
            )
        return None
    
    def check_food_quality_alert(self, quality_score: float) -> Alert:
        """
        Check if food quality warrants an alert
        """
        if quality_score < self.threshold_config['food_quality_critical']:
            return self.create_alert(
                AlertType.FOOD_QUALITY,
                AlertSeverity.HIGH,
                f"CRITICAL: Food quality issue reported (Score: {quality_score})",
                {'quality_score': quality_score}
            )
        return None
    
    def check_room_quality_alert(self, room_id: str, cleanliness_score: float) -> Alert:
        """
        Check if room quality warrants an alert
        """
        if cleanliness_score < self.threshold_config['room_quality_critical']:
            return self.create_alert(
                AlertType.ROOM_QUALITY,
                AlertSeverity.HIGH,
                f"CRITICAL: Room {room_id} cleanliness issue (Score: {cleanliness_score})",
                {'room_id': room_id, 'cleanliness_score': cleanliness_score}
            )
        return None
    
    def check_issue_cluster_alert(self, cluster_data: Dict) -> Alert:
        """
        Check if issue cluster is significant enough for alert
        """
        if cluster_data.get('frequency', 0) >= self.threshold_config['issue_cluster_critical']:
            return self.create_alert(
                AlertType.ISSUE_CLUSTER,
                AlertSeverity.MEDIUM,
                f"Issue cluster detected: {cluster_data.get('primary_issue')} ({cluster_data.get('frequency')} cases)",
                cluster_data
            )
        return None
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Dict]:
        """
        Get active (unacknowledged) alerts
        """
        active_alerts = [a for a in self.alerts if not a.acknowledged]
        
        if severity:
            active_alerts = [a for a in active_alerts if a.severity == severity]
        
        return [a.to_dict() for a in sorted(
            active_alerts,
            key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}[x.severity.value],
            reverse=True
        )]
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """
        Acknowledge an alert
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge(user_id)
                return True
        return False
    
    def get_alert_statistics(self, hours: int = 24) -> Dict:
        """
        Get alert statistics
        """
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a.created_at) > cutoff_time
        ]
        
        severity_count = {
            'CRITICAL': sum(1 for a in recent_alerts if a.severity == AlertSeverity.CRITICAL),
            'HIGH': sum(1 for a in recent_alerts if a.severity == AlertSeverity.HIGH),
            'MEDIUM': sum(1 for a in recent_alerts if a.severity == AlertSeverity.MEDIUM),
            'LOW': sum(1 for a in recent_alerts if a.severity == AlertSeverity.LOW)
        }
        
        type_count = {}
        for alert in recent_alerts:
            alert_type = alert.alert_type.value
            type_count[alert_type] = type_count.get(alert_type, 0) + 1
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'unacknowledged_alerts': sum(1 for a in recent_alerts if not a.acknowledged),
            'severity_distribution': severity_count,
            'type_distribution': type_count
        }
    
    def export_alerts_json(self, filepath: str, hours: int = 24) -> bool:
        """
        Export alerts to JSON file
        """
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            alerts_to_export = [
                a.to_dict() for a in self.alerts
                if datetime.fromisoformat(a.created_at) > cutoff_time
            ]
            
            with open(filepath, 'w') as f:
                json.dump({
                    'export_time': datetime.now().isoformat(),
                    'period_hours': hours,
                    'total_alerts': len(alerts_to_export),
                    'alerts': alerts_to_export
                }, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting alerts: {e}")
            return False


# Initialize global alert manager
alert_manager = AlertManager()


# Example alert handlers
def email_alert_handler(alert: Alert) -> None:
    """
    Example handler to send email for critical alerts
    In production, integrate with email service
    """
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
        print(f"[EMAIL] Sending alert notification: {alert.message}")


def log_alert_handler(alert: Alert) -> None:
    """
    Example handler to log alerts
    """
    print(f"[LOG] Alert {alert.alert_id}: {alert.message}")


# Register handlers
alert_manager.register_handler(AlertType.SENTIMENT_CRITICAL, email_alert_handler)
alert_manager.register_handler(AlertType.DOCTOR_BURNOUT, email_alert_handler)
alert_manager.register_handler(AlertType.COMPLAINT_FILED, email_alert_handler)

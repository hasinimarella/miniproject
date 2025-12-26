from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


class DoctorAnalyzer:
    """
    Analyzes doctor performance, burnout risk, and complaint patterns
    """
    
    def __init__(self):
        self.doctors = {}  # In-memory storage (replace with DB in production)
        self.duty_logs = defaultdict(list)
        self.complaints = defaultdict(list)
        self.sentiment_scores = defaultdict(list)
    
    def register_duty_shift(self, doctor_id: str, shift_date: str, hours: float, 
                           patient_count: int, emergency_cases: int = 0) -> Dict:
        """
        Log a doctor's duty shift
        """
        shift_info = {
            'date': shift_date,
            'hours': hours,
            'patient_count': patient_count,
            'emergency_cases': emergency_cases,
            'workload_index': (patient_count / hours) if hours > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.duty_logs[doctor_id].append(shift_info)
        
        return {
            'status': 'logged',
            'doctor_id': doctor_id,
            'shift_info': shift_info
        }
    
    def calculate_burnout_risk(self, doctor_id: str, days_window: int = 30) -> Dict:
        """
        Calculate burnout risk based on workload metrics
        Risk factors:
        - Average daily hours
        - Patient load
        - Emergency cases
        - Consecutive shifts without break
        """
        recent_shifts = self.duty_logs.get(doctor_id, [])
        
        if not recent_shifts:
            return {
                'doctor_id': doctor_id,
                'risk_level': 'NO_DATA',
                'risk_score': 0,
                'metrics': {}
            }
        
        # Filter to recent period
        cutoff_date = datetime.now() - timedelta(days=days_window)
        recent = [
            s for s in recent_shifts 
            if datetime.fromisoformat(s['timestamp']) > cutoff_date
        ]
        
        if not recent:
            return {
                'doctor_id': doctor_id,
                'risk_level': 'LOW',
                'risk_score': 0,
                'metrics': {}
            }
        
        # Calculate metrics
        total_hours = sum(s['hours'] for s in recent)
        total_patients = sum(s['patient_count'] for s in recent)
        total_emergencies = sum(s['emergency_cases'] for s in recent)
        num_shifts = len(recent)
        
        avg_hours_per_shift = total_hours / num_shifts if num_shifts > 0 else 0
        avg_patients_per_shift = total_patients / num_shifts if num_shifts > 0 else 0
        
        # Calculate burnout score (0-1)
        # Factors weighted by importance
        hours_factor = min(avg_hours_per_shift / 12, 1.0) * 0.4  # High > 12 hrs/shift
        patient_factor = min(avg_patients_per_shift / 30, 1.0) * 0.35  # High > 30 patients/shift
        emergency_factor = min(total_emergencies / num_shifts / 5, 1.0) * 0.25  # High > 5 emergencies/shift
        
        burnout_score = hours_factor + patient_factor + emergency_factor
        
        # Determine risk level
        if burnout_score > 0.7:
            risk_level = 'CRITICAL'
        elif burnout_score > 0.5:
            risk_level = 'HIGH'
        elif burnout_score > 0.3:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'doctor_id': doctor_id,
            'risk_level': risk_level,
            'risk_score': round(burnout_score, 3),
            'metrics': {
                'average_hours_per_shift': round(avg_hours_per_shift, 2),
                'average_patients_per_shift': round(avg_patients_per_shift, 2),
                'total_emergency_cases': total_emergencies,
                'total_shifts_analyzed': num_shifts,
                'hours_factor': round(hours_factor, 3),
                'patient_factor': round(patient_factor, 3),
                'emergency_factor': round(emergency_factor, 3)
            },
            'recommendations': self._get_burnout_recommendations(burnout_score)
        }
    
    def _get_burnout_recommendations(self, burnout_score: float) -> List[str]:
        """
        Generate recommendations based on burnout score
        """
        recommendations = []
        
        if burnout_score > 0.7:
            recommendations.extend([
                "URGENT: Reduce work schedule immediately",
                "Schedule wellness check-up",
                "Consider temporary leave",
                "Assign additional support staff"
            ])
        elif burnout_score > 0.5:
            recommendations.extend([
                "Monitor workload closely",
                "Reduce patient load by 20-30%",
                "Encourage breaks and time off",
                "Provide mental health support"
            ])
        elif burnout_score > 0.3:
            recommendations.extend([
                "Continue current schedule with monitoring",
                "Encourage regular breaks",
                "Schedule monthly wellness check-ins"
            ])
        else:
            recommendations.append("Current workload is sustainable")
        
        return recommendations
    
    def file_complaint(self, doctor_id: str, complaint_type: str, 
                      description: str, severity: str = 'MEDIUM') -> Dict:
        """
        File a complaint against a doctor
        """
        complaint = {
            'complaint_id': f"CMPL_{doctor_id}_{datetime.now().timestamp()}",
            'doctor_id': doctor_id,
            'type': complaint_type,
            'description': description,
            'severity': severity,
            'status': 'OPEN',
            'filed_date': datetime.now().isoformat(),
            'resolved_date': None
        }
        
        self.complaints[doctor_id].append(complaint)
        
        return {
            'status': 'filed',
            'complaint_id': complaint['complaint_id'],
            'doctor_id': doctor_id
        }
    
    def get_complaint_history(self, doctor_id: str) -> Dict:
        """
        Get complaint history and patterns for a doctor
        """
        complaints = self.complaints.get(doctor_id, [])
        
        if not complaints:
            return {
                'doctor_id': doctor_id,
                'total_complaints': 0,
                'complaint_patterns': {}
            }
        
        # Analyze complaint patterns
        complaint_types = defaultdict(int)
        severity_dist = defaultdict(int)
        resolution_rate = 0
        
        for complaint in complaints:
            complaint_types[complaint['type']] += 1
            severity_dist[complaint['severity']] += 1
            if complaint['status'] == 'RESOLVED':
                resolution_rate += 1
        
        resolution_rate = (resolution_rate / len(complaints) * 100) if complaints else 0
        
        return {
            'doctor_id': doctor_id,
            'total_complaints': len(complaints),
            'complaint_patterns': dict(complaint_types),
            'severity_distribution': dict(severity_dist),
            'resolution_rate': round(resolution_rate, 2),
            'recent_complaints': complaints[-5:],  # Last 5 complaints
            'complaint_status': self._calculate_complaint_status(complaints)
        }
    
    def _calculate_complaint_status(self, complaints: List[Dict]) -> str:
        """
        Calculate overall complaint status
        """
        if not complaints:
            return 'GOOD'
        
        unresolved = sum(1 for c in complaints if c['status'] == 'OPEN')
        critical = sum(1 for c in complaints if c['severity'] == 'CRITICAL')
        
        if critical > 0 or unresolved > 3:
            return 'CRITICAL'
        elif unresolved > 1:
            return 'CONCERNING'
        else:
            return 'ACCEPTABLE'
    
    def track_sentiment(self, doctor_id: str, sentiment_score: float, 
                       review_id: str = None) -> None:
        """
        Track patient sentiment about a doctor
        """
        self.sentiment_scores[doctor_id].append({
            'score': sentiment_score,
            'review_id': review_id,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_doctor_rating(self, doctor_id: str) -> Dict:
        """
        Calculate overall doctor rating based on sentiment
        """
        sentiments = self.sentiment_scores.get(doctor_id, [])
        
        if not sentiments:
            return {
                'doctor_id': doctor_id,
                'rating': 0,
                'total_reviews': 0,
                'rating_distribution': {}
            }
        
        scores = [s['score'] for s in sentiments]
        avg_score = np.mean(scores)
        
        # Convert sentiment score (-1 to 1) to rating (1 to 5)
        rating = ((avg_score + 1) / 2) * 4 + 1
        
        # Distribution
        rating_dist = defaultdict(int)
        for score in scores:
            rating_val = ((score + 1) / 2) * 4 + 1
            bin_num = min(5, max(1, int(rating_val)))
            rating_dist[bin_num] += 1
        
        return {
            'doctor_id': doctor_id,
            'rating': round(rating, 2),
            'total_reviews': len(sentiments),
            'average_sentiment_score': round(avg_score, 3),
            'rating_distribution': dict(rating_dist),
            'status': self._get_rating_status(rating)
        }
    
    def _get_rating_status(self, rating: float) -> str:
        """
        Get status based on rating
        """
        if rating >= 4.5:
            return 'EXCELLENT'
        elif rating >= 4.0:
            return 'VERY_GOOD'
        elif rating >= 3.5:
            return 'GOOD'
        elif rating >= 3.0:
            return 'FAIR'
        else:
            return 'POOR'
    
    def get_doctor_performance_dashboard(self, doctor_id: str) -> Dict:
        """
        Comprehensive performance dashboard for a doctor
        """
        burnout = self.calculate_burnout_risk(doctor_id)
        complaints = self.get_complaint_history(doctor_id)
        rating = self.get_doctor_rating(doctor_id)
        
        # Calculate overall performance score
        performance_score = (
            (rating['rating'] / 5 * 100) * 0.5 +
            ((1 - burnout['risk_score']) * 100) * 0.3 +
            (100 - min(complaints['total_complaints'] * 5, 100)) * 0.2
        )
        
        return {
            'doctor_id': doctor_id,
            'overall_performance_score': round(performance_score, 2),
            'burnout_analysis': burnout,
            'complaint_analysis': complaints,
            'patient_rating': rating,
            'dashboard_generated': datetime.now().isoformat()
        }


# Initialize global analyzer instance
doctor_analyzer = DoctorAnalyzer()

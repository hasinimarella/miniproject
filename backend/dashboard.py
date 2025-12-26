from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class SentimentTrendAnalyzer:
    """
    Analyzes sentiment trends and clusters issues
    """
    
    def __init__(self):
        self.reviews = []
        self.issue_clusters = []
    
    def add_review_analysis(
        self,
        review_id: str,
        sentiment_analysis: Dict,
        category: str = 'general',
        metadata: Dict = None
    ) -> None:
        """
        Add a analyzed review for trend tracking
        """
        review_data = {
            'review_id': review_id,
            'sentiment_score': sentiment_analysis['overall_score'],
            'sentiment_label': sentiment_analysis['sentiment_label'],
            'category': category,
            'emotions': sentiment_analysis['emotions'],
            'keywords': sentiment_analysis['keywords'],
            'original_language': sentiment_analysis['original_language'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Optional contextual metadata to improve analytics
        if metadata:
            review_data.update({
                'input_method': metadata.get('input_method'),
                'patient_id': metadata.get('patient_id'),
                'patient_name': metadata.get('patient_name'),
                'rating': metadata.get('rating'),
                'language': metadata.get('language')
            })
        
        self.reviews.append(review_data)
    
    def get_sentiment_trends(self, days: int = 30, category: str = None) -> Dict:
        """
        Analyze sentiment trends over a period
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_reviews = [
            r for r in self.reviews
            if datetime.fromisoformat(r['timestamp']) > cutoff_date
            and (category is None or r['category'] == category)
        ]
        
        if not relevant_reviews:
            return {
                'period_days': days,
                'category': category,
                'total_reviews': 0,
                'trend_data': []
            }
        
        # Group by date
        daily_trends = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'avg_score': []})
        
        for review in relevant_reviews:
            date_str = review['timestamp'].split('T')[0]
            sentiment = review['sentiment_label']
            
            if sentiment == 'POSITIVE':
                daily_trends[date_str]['positive'] += 1
            elif sentiment == 'NEGATIVE':
                daily_trends[date_str]['negative'] += 1
            else:
                daily_trends[date_str]['neutral'] += 1
            
            daily_trends[date_str]['avg_score'].append(review['sentiment_score'])
        
        # Calculate averages
        trend_data = []
        for date in sorted(daily_trends.keys()):
            day_data = daily_trends[date]
            trend_data.append({
                'date': date,
                'positive': day_data['positive'],
                'negative': day_data['negative'],
                'neutral': day_data['neutral'],
                'average_score': round(np.mean(day_data['avg_score']), 3),
                'total_reviews': day_data['positive'] + day_data['negative'] + day_data['neutral']
            })
        
        return {
            'period_days': days,
            'category': category,
            'total_reviews': len(relevant_reviews),
            'trend_data': trend_data,
            'overall_average_score': round(np.mean([r['sentiment_score'] for r in relevant_reviews]), 3),
            'sentiment_distribution': self._calculate_distribution(relevant_reviews)
        }
    
    def _calculate_distribution(self, reviews: List[Dict]) -> Dict:
        """
        Calculate sentiment distribution
        """
        positive = sum(1 for r in reviews if r['sentiment_label'] == 'POSITIVE')
        negative = sum(1 for r in reviews if r['sentiment_label'] == 'NEGATIVE')
        neutral = sum(1 for r in reviews if r['sentiment_label'] == 'NEUTRAL')
        total = len(reviews)
        
        return {
            'positive_percentage': round(positive / total * 100, 2) if total > 0 else 0,
            'negative_percentage': round(negative / total * 100, 2) if total > 0 else 0,
            'neutral_percentage': round(neutral / total * 100, 2) if total > 0 else 0
        }
    
    def cluster_issues(self, max_clusters: int = 5) -> Dict:
        """
        Cluster similar issues using text analysis
        """
        if len(self.reviews) < 3:
            return {
                'clusters': [],
                'clustering_method': 'keyword_based'
            }
        
        # Extract keywords from negative reviews
        negative_reviews = [r for r in self.reviews if r['sentiment_label'] == 'NEGATIVE']
        
        if not negative_reviews:
            return {
                'clusters': [],
                'total_reviews': len(self.reviews),
                'message': 'No negative reviews to cluster'
            }
        
        # Collect all keywords
        all_keywords = defaultdict(int)
        review_keyword_map = {}
        
        for review in negative_reviews:
            review_keywords = review.get('keywords', [])
            review_keyword_map[review['review_id']] = review_keywords
            for keyword in review_keywords:
                all_keywords[keyword] += 1
        
        # Create issue clusters based on keyword frequency
        issue_clusters = []
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        
        for idx, (keyword, count) in enumerate(sorted_keywords[:max_clusters]):
            # Find reviews with this keyword
            related_reviews = [
                rid for rid, keywords in review_keyword_map.items()
                if keyword in keywords
            ]
            
            issue_clusters.append({
                'cluster_id': idx + 1,
                'primary_issue': keyword,
                'frequency': count,
                'related_reviews': related_reviews,
                'severity': self._calculate_cluster_severity(related_reviews),
                'recommendation': self._get_cluster_recommendation(keyword)
            })
        
        return {
            'total_negative_reviews': len(negative_reviews),
            'clusters': issue_clusters,
            'clustering_method': 'keyword_frequency_based'
        }
    
    def _calculate_cluster_severity(self, review_ids: List[str]) -> str:
        """
        Calculate severity of an issue cluster
        """
        if len(review_ids) >= 10:
            return 'CRITICAL'
        elif len(review_ids) >= 5:
            return 'HIGH'
        elif len(review_ids) >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_cluster_recommendation(self, issue: str) -> str:
        """
        Get recommendations based on issue type
        """
        issue_lower = issue.lower()
        
        if any(word in issue_lower for word in ['staff', 'doctor', 'nurse']):
            return 'Review staff training and conduct 1-on-1 meetings'
        elif any(word in issue_lower for word in ['clean', 'dirty', 'hygiene']):
            return 'Increase housekeeping frequency and conduct audits'
        elif any(word in issue_lower for word in ['food', 'meal', 'diet']):
            return 'Review food preparation procedures with catering team'
        elif any(word in issue_lower for word in ['wait', 'delay', 'slow']):
            return 'Review process efficiency and resource allocation'
        elif any(word in issue_lower for word in ['pain', 'discomfort', 'ache']):
            return 'Review pain management protocols with medical team'
        else:
            return 'Investigate and implement corrective measures'
    
    def get_emotion_analysis(self, category: str = None, days: int = 30) -> Dict:
        """
        Analyze emotion distribution across reviews
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_reviews = [
            r for r in self.reviews
            if datetime.fromisoformat(r['timestamp']) > cutoff_date
            and (category is None or r['category'] == category)
        ]
        
        if not relevant_reviews:
            return {
                'period_days': days,
                'total_reviews': 0,
                'emotion_distribution': {}
            }
        
        # Aggregate emotions
        emotion_totals = defaultdict(float)
        
        for review in relevant_reviews:
            for emotion, score in review.get('emotions', {}).items():
                emotion_totals[emotion] += score
        
        # Calculate averages
        emotion_distribution = {
            emotion: round(total / len(relevant_reviews), 3)
            for emotion, total in emotion_totals.items()
        }
        
        return {
            'period_days': days,
            'category': category,
            'total_reviews': len(relevant_reviews),
            'emotion_distribution': emotion_distribution,
            'dominant_emotion': max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else None
        }


class OperationsDashboard:
    """
    Comprehensive hospital operations dashboard
    """
    
    def __init__(self, sentiment_analyzer, doctor_analyzer, facility_monitor):
        self.sentiment_analyzer = sentiment_analyzer
        self.doctor_analyzer = doctor_analyzer
        self.facility_monitor = facility_monitor
    
    def get_dashboard_overview(self) -> Dict:
        """
        Get comprehensive dashboard overview
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'sentiment_summary': self._get_sentiment_summary(),
            'doctor_summary': self._get_doctor_summary(),
            'facility_summary': self._get_facility_summary(),
            'critical_alerts': self._get_critical_alerts(),
            'key_metrics': self._calculate_key_metrics()
        }
    
    def _get_sentiment_summary(self) -> Dict:
        """
        Get sentiment summary from trend analyzer
        """
        trends = self.sentiment_analyzer.get_sentiment_trends(days=7)
        
        return {
            'total_reviews_week': trends['total_reviews'],
            'average_sentiment_score': trends.get('overall_average_score', 0),
            'distribution': trends.get('sentiment_distribution', {})
        }
    
    def _get_doctor_summary(self) -> Dict:
        """
        Get doctor analytics summary
        """
        return {
            'total_complaints': len(self.doctor_analyzer.complaints),
            'open_complaints': sum(
                1 for complaints in self.doctor_analyzer.complaints.values()
                for c in complaints if c['status'] == 'OPEN'
            ),
            'burnout_risk_doctors': sum(
                1 for doc_id in self.doctor_analyzer.doctors
                if self.doctor_analyzer.calculate_burnout_risk(doc_id)['risk_level'] in ['HIGH', 'CRITICAL']
            )
        }
    
    def _get_facility_summary(self) -> Dict:
        """
        Get facility quality summary
        """
        food_analysis = self.facility_monitor.analyze_food_quality_trends()
        room_analysis = self.facility_monitor.analyze_room_quality_trends()
        
        return {
            'food_quality_score': food_analysis.get('average_quality_score', 0),
            'room_cleanliness_score': room_analysis.get('average_cleanliness_score', 0),
            'problem_rooms': len(room_analysis.get('problem_rooms', []))
        }
    
    def _get_critical_alerts(self) -> List[Dict]:
        """
        Get critical alerts from all systems
        """
        alerts = []
        
        # Check for critical burnout cases
        for doc_id, _ in self.doctor_analyzer.doctors.items():
            burnout = self.doctor_analyzer.calculate_burnout_risk(doc_id)
            if burnout['risk_level'] == 'CRITICAL':
                alerts.append({
                    'type': 'BURNOUT_CRITICAL',
                    'doctor_id': doc_id,
                    'severity': 'HIGH',
                    'message': f'Doctor {doc_id} at critical burnout level'
                })
        
        # Check for critical complaints
        for doc_id, complaints in self.doctor_analyzer.complaints.items():
            critical_complaints = [c for c in complaints if c['severity'] == 'CRITICAL']
            if critical_complaints:
                alerts.append({
                    'type': 'CRITICAL_COMPLAINT',
                    'doctor_id': doc_id,
                    'severity': 'CRITICAL',
                    'message': f'{len(critical_complaints)} critical complaint(s) for doctor {doc_id}'
                })
        
        return sorted(alerts, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}[x['severity']])
    
    def _calculate_key_metrics(self) -> Dict:
        """
        Calculate key performance metrics
        """
        return {
            'system_health': 'OPERATIONAL',
            'data_freshness': 'CURRENT',
            'alert_count': len(self._get_critical_alerts()),
            'last_update': datetime.now().isoformat()
        }


# Initialize global instances
trend_analyzer = SentimentTrendAnalyzer()

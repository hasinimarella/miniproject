from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class FacilityQualityMonitor:
    """
    Monitors and analyzes food quality, room cleanliness, and overall facility conditions
    """
    
    def __init__(self):
        self.food_reviews = defaultdict(list)
        self.room_reviews = defaultdict(list)
        self.facility_ratings = {}
    
    def submit_food_quality_review(self, review_id: str, quality_score: float,
                                  aspects: Dict[str, float], comments: str = "") -> Dict:
        """
        Submit food quality review
        Aspects: taste, hygiene, temperature, variety, portion_size
        Quality_score: 1-5 scale
        """
        review = {
            'review_id': review_id,
            'quality_score': quality_score,
            'aspects': aspects,
            'comments': comments,
            'submitted_date': datetime.now().isoformat(),
            'status': 'ACTIVE'
        }
        
        self.food_reviews[review_id].append(review)
        
        # Trigger alert if score is very low
        alert = None
        if quality_score <= 2:
            alert = {
                'type': 'FOOD_QUALITY_CRITICAL',
                'severity': 'HIGH',
                'message': f'Critical food quality issue reported (Score: {quality_score})',
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'status': 'submitted',
            'review_id': review_id,
            'alert': alert
        }
    
    def submit_room_quality_review(self, review_id: str, room_id: str, 
                                  cleanliness_score: float, aspects: Dict[str, float],
                                  comments: str = "") -> Dict:
        """
        Submit room quality review
        Aspects: cleanliness, furniture_condition, bathroom_condition, bed_comfort, lighting
        Cleanliness_score: 1-5 scale
        """
        review = {
            'review_id': review_id,
            'room_id': room_id,
            'cleanliness_score': cleanliness_score,
            'aspects': aspects,
            'comments': comments,
            'submitted_date': datetime.now().isoformat(),
            'status': 'ACTIVE'
        }
        
        self.room_reviews[room_id].append(review)
        
        # Trigger alert if score is very low
        alert = None
        if cleanliness_score <= 2:
            alert = {
                'type': 'ROOM_QUALITY_CRITICAL',
                'severity': 'HIGH',
                'room_id': room_id,
                'message': f'Critical room cleanliness issue in room {room_id} (Score: {cleanliness_score})',
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'status': 'submitted',
            'review_id': review_id,
            'room_id': room_id,
            'alert': alert
        }
    
    def analyze_food_quality_trends(self, days: int = 30) -> Dict:
        """
        Analyze food quality trends over a period
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        all_reviews = []
        for review_list in self.food_reviews.values():
            for review in review_list:
                if datetime.fromisoformat(review['submitted_date']) > cutoff_date:
                    all_reviews.append(review)
        
        if not all_reviews:
            return {
                'period_days': days,
                'total_reviews': 0,
                'average_quality_score': 0,
                'quality_distribution': {}
            }
        
        scores = [r['quality_score'] for r in all_reviews]
        avg_score = np.mean(scores)
        
        # Aspect analysis
        all_aspects = {}
        for review in all_reviews:
            for aspect, score in review['aspects'].items():
                if aspect not in all_aspects:
                    all_aspects[aspect] = []
                all_aspects[aspect].append(score)
        
        aspect_averages = {
            aspect: round(np.mean(scores), 2)
            for aspect, scores in all_aspects.items()
        }
        
        # Quality distribution
        quality_dist = defaultdict(int)
        for score in scores:
            bin_num = int(score) if score < 5 else 5
            quality_dist[bin_num] += 1
        
        return {
            'period_days': days,
            'total_reviews': len(all_reviews),
            'average_quality_score': round(avg_score, 2),
            'quality_distribution': dict(quality_dist),
            'aspect_analysis': aspect_averages,
            'quality_status': self._get_food_quality_status(avg_score),
            'improvement_needed': [
                aspect for aspect, score in aspect_averages.items() if score < 3
            ]
        }
    
    def analyze_room_quality_trends(self, days: int = 30) -> Dict:
        """
        Analyze room quality trends and identify problem areas
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        all_reviews = []
        for review_list in self.room_reviews.values():
            for review in review_list:
                if datetime.fromisoformat(review['submitted_date']) > cutoff_date:
                    all_reviews.append(review)
        
        if not all_reviews:
            return {
                'period_days': days,
                'total_reviews': 0,
                'average_cleanliness_score': 0,
                'problem_rooms': []
            }
        
        scores = [r['cleanliness_score'] for r in all_reviews]
        avg_score = np.mean(scores)
        
        # Room-wise analysis
        room_scores = defaultdict(list)
        for review in all_reviews:
            room_scores[review['room_id']].append(review['cleanliness_score'])
        
        room_averages = {
            room: round(np.mean(room_scores_list), 2)
            for room, room_scores_list in room_scores.items()
        }
        
        # Aspect analysis
        all_aspects = {}
        for review in all_reviews:
            for aspect, score in review['aspects'].items():
                if aspect not in all_aspects:
                    all_aspects[aspect] = []
                all_aspects[aspect].append(score)
        
        aspect_averages = {
            aspect: round(np.mean(scores), 2)
            for aspect, scores in all_aspects.items()
        }
        
        # Identify problem rooms
        problem_rooms = [
            {'room_id': room, 'score': score}
            for room, score in room_averages.items() if score < 3
        ]
        
        return {
            'period_days': days,
            'total_reviews': len(all_reviews),
            'average_cleanliness_score': round(avg_score, 2),
            'room_wise_scores': room_averages,
            'aspect_analysis': aspect_averages,
            'problem_rooms': sorted(problem_rooms, key=lambda x: x['score']),
            'quality_status': self._get_room_quality_status(avg_score),
            'maintenance_priority': self._get_maintenance_priority(room_averages)
        }
    
    def _get_food_quality_status(self, avg_score: float) -> str:
        """
        Get food quality status
        """
        if avg_score >= 4.5:
            return 'EXCELLENT'
        elif avg_score >= 4.0:
            return 'VERY_GOOD'
        elif avg_score >= 3.5:
            return 'GOOD'
        elif avg_score >= 2.5:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _get_room_quality_status(self, avg_score: float) -> str:
        """
        Get room quality status
        """
        if avg_score >= 4.5:
            return 'EXCELLENT'
        elif avg_score >= 4.0:
            return 'VERY_GOOD'
        elif avg_score >= 3.5:
            return 'GOOD'
        elif avg_score >= 2.5:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _get_maintenance_priority(self, room_scores: Dict[str, float]) -> List[Dict]:
        """
        Get maintenance priority list
        """
        priority_list = []
        for room_id, score in room_scores.items():
            if score < 3:
                priority = 'URGENT'
            elif score < 3.5:
                priority = 'HIGH'
            elif score < 4:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            priority_list.append({
                'room_id': room_id,
                'score': score,
                'priority': priority
            })
        
        return sorted(priority_list, key=lambda x: {
            'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3
        }[x['priority']])
    
    def get_food_quality_recommendations(self, days: int = 30) -> List[str]:
        """
        Get recommendations for food quality improvement
        """
        analysis = self.analyze_food_quality_trends(days)
        recommendations = []
        
        if analysis['average_quality_score'] < 3:
            recommendations.append("URGENT: Food quality is below acceptable standards")
            recommendations.append("Review food preparation procedures")
            recommendations.append("Conduct food safety audit")
        
        for aspect in analysis.get('improvement_needed', []):
            recommendations.append(f"Improve {aspect} quality")
        
        return recommendations
    
    def get_room_quality_recommendations(self, days: int = 30) -> List[str]:
        """
        Get recommendations for room quality improvement
        """
        analysis = self.analyze_room_quality_trends(days)
        recommendations = []
        
        if analysis['average_cleanliness_score'] < 3:
            recommendations.append("URGENT: Room cleanliness is below acceptable standards")
            recommendations.append("Increase housekeeping staff")
            recommendations.append("Conduct training for housekeeping team")
        
        for room in analysis.get('problem_rooms', [])[:5]:
            recommendations.append(f"Priority: Deep clean room {room['room_id']}")
        
        return recommendations


# Initialize global monitor instance
facility_monitor = FacilityQualityMonitor()

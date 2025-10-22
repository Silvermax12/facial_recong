import time
from datetime import datetime
from typing import Dict, Any


class AdaptiveThresholdManager:
    """
    Dynamic threshold adjustment based on risk context
    Implements risk-based adaptive thresholds for liveness detection
    """
    
    def __init__(self, base_threshold=0.75):
        self.base_threshold = base_threshold
        self.min_threshold = 0.60
        self.max_threshold = 0.95
    
    def get_threshold(self, context: Dict[str, Any]) -> float:
        """
        Calculate dynamic threshold based on risk factors
        
        Args:
            context: Dictionary containing risk context:
                - device_trusted: bool
                - device_rooted: bool
                - user_login_count: int
                - recent_failed_attempts: int
                - timestamp: float (unix timestamp)
                - user_timezone: str
                - transaction_amount: float (optional)
                - ip_reputation: float (0-1, optional)
                - session_duration: float (seconds, optional)
        
        Returns:
            Adjusted threshold value (0.60 - 0.95)
        """
        threshold = self.base_threshold
        
        # Device trust factor
        if context.get('device_trusted', False):
            threshold -= 0.05  # Trusted device, lower threshold
        
        if context.get('device_rooted', False):
            threshold += 0.10  # Rooted device, higher threshold (security risk)
        
        # User history and behavior
        login_count = context.get('user_login_count', 0)
        if login_count > 100:
            threshold -= 0.05  # Established user, lower threshold
        elif login_count < 5:
            threshold += 0.05  # New user, higher threshold
        
        # Failed attempt tracking
        failed_attempts = context.get('recent_failed_attempts', 0)
        if failed_attempts > 3:
            threshold += 0.15  # Multiple failures, much higher threshold
        elif failed_attempts > 1:
            threshold += 0.08  # Some failures, moderately higher
        
        # Transaction value (if applicable)
        transaction_amount = context.get('transaction_amount', 0)
        if transaction_amount > 10000:
            threshold += 0.10  # High-value transaction, higher security
        elif transaction_amount > 1000:
            threshold += 0.05  # Medium-value transaction
        
        # Time-based risk factors
        if self._is_unusual_time(context):
            threshold += 0.05  # Unusual time, increase threshold
        
        # IP reputation (if available)
        ip_reputation = context.get('ip_reputation', 1.0)
        if ip_reputation < 0.5:
            threshold += 0.10  # Suspicious IP, higher threshold
        
        # Session duration (rapid attempts are suspicious)
        session_duration = context.get('session_duration', float('inf'))
        if session_duration < 5:  # Less than 5 seconds
            threshold += 0.10  # Very fast attempt, suspicious
        
        # Clamp to valid range
        return min(max(threshold, self.min_threshold), self.max_threshold)
    
    def _is_unusual_time(self, context: Dict[str, Any]) -> bool:
        """
        Check if verification is happening at unusual time
        
        Args:
            context: Must contain 'timestamp' and optionally 'user_timezone'
        
        Returns:
            True if time is unusual (late night/early morning)
        """
        timestamp = context.get('timestamp')
        if timestamp is None:
            return False
        
        # Convert to datetime
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        
        # Consider 1 AM - 5 AM as unusual (local time)
        # In production, this should use user_timezone
        return 1 <= hour <= 5
    
    def get_risk_level(self, context: Dict[str, Any]) -> str:
        """
        Categorize risk level based on context
        
        Args:
            context: Risk context dictionary
        
        Returns:
            Risk level: 'low', 'medium', 'high', 'critical'
        """
        threshold = self.get_threshold(context)
        
        if threshold >= 0.90:
            return 'critical'
        elif threshold >= 0.80:
            return 'high'
        elif threshold >= 0.70:
            return 'medium'
        else:
            return 'low'
    
    def get_threshold_explanation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed explanation of threshold calculation
        
        Args:
            context: Risk context dictionary
        
        Returns:
            Dictionary with threshold and contributing factors
        """
        factors = []
        adjustments = []
        
        base = self.base_threshold
        current = base
        
        # Track all adjustments
        if context.get('device_trusted', False):
            current -= 0.05
            factors.append('device_trusted')
            adjustments.append(-0.05)
        
        if context.get('device_rooted', False):
            current += 0.10
            factors.append('device_rooted')
            adjustments.append(+0.10)
        
        login_count = context.get('user_login_count', 0)
        if login_count > 100:
            current -= 0.05
            factors.append('established_user')
            adjustments.append(-0.05)
        elif login_count < 5:
            current += 0.05
            factors.append('new_user')
            adjustments.append(+0.05)
        
        failed_attempts = context.get('recent_failed_attempts', 0)
        if failed_attempts > 3:
            current += 0.15
            factors.append('multiple_failed_attempts')
            adjustments.append(+0.15)
        elif failed_attempts > 1:
            current += 0.08
            factors.append('some_failed_attempts')
            adjustments.append(+0.08)
        
        transaction_amount = context.get('transaction_amount', 0)
        if transaction_amount > 10000:
            current += 0.10
            factors.append('high_value_transaction')
            adjustments.append(+0.10)
        elif transaction_amount > 1000:
            current += 0.05
            factors.append('medium_value_transaction')
            adjustments.append(+0.05)
        
        if self._is_unusual_time(context):
            current += 0.05
            factors.append('unusual_time')
            adjustments.append(+0.05)
        
        ip_reputation = context.get('ip_reputation')
        if ip_reputation is not None and ip_reputation < 0.5:
            current += 0.10
            factors.append('suspicious_ip')
            adjustments.append(+0.10)
        
        session_duration = context.get('session_duration')
        if session_duration is not None and session_duration < 5:
            current += 0.10
            factors.append('rapid_attempt')
            adjustments.append(+0.10)
        
        # Final clamped value
        final_threshold = min(max(current, self.min_threshold), self.max_threshold)
        
        return {
            'base_threshold': base,
            'calculated_threshold': current,
            'final_threshold': final_threshold,
            'risk_level': self.get_risk_level(context),
            'contributing_factors': factors,
            'adjustments': adjustments,
            'total_adjustment': sum(adjustments)
        }


def get_user_risk_profile(username: str, user_database) -> Dict[str, Any]:
    """
    Get user risk profile from database (example function)
    
    Args:
        username: Username to lookup
        user_database: Database connection or user store
    
    Returns:
        User risk profile dictionary
    """
    # This is a placeholder - implement based on your user database
    # Example return:
    return {
        'user_login_count': 0,
        'recent_failed_attempts': 0,
        'last_login_timestamp': None,
        'trusted_devices': [],
        'average_session_duration': 0
    }


"""
Verification Logging and Suspicious Activity Tracking
Comprehensive logging system for facial recognition and liveness detection
"""

import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import threading


class VerificationLogger:
    """
    Comprehensive logging system for verification attempts
    Tracks all attempts, failures, and suspicious activity patterns
    """
    
    def __init__(self, log_path: str = "logs/verification.log", 
                 retention_days: int = 30,
                 enable_suspicious_tracking: bool = True):
        self.log_path = log_path
        self.retention_days = retention_days
        self.enable_suspicious_tracking = enable_suspicious_tracking
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else "logs", exist_ok=True)
        
        # Thread-safe logging
        self.lock = threading.Lock()
        
        # In-memory suspicious activity tracking
        self.failed_attempts = defaultdict(list)  # username -> [timestamps]
        self.ip_attempts = defaultdict(list)      # ip -> [timestamps]
        self.suspicious_patterns = defaultdict(int)  # pattern -> count
        
    def log_attempt(self, 
                   username: str,
                   success: bool,
                   liveness_score: float,
                   quality_score: float,
                   reason: Optional[str] = None,
                   ip_address: Optional[str] = None,
                   metadata: Optional[Dict] = None):
        """Log a verification attempt"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "success": success,
            "liveness_score": liveness_score,
            "quality_score": quality_score,
            "reason": reason,
            "ip_address": ip_address,
            "metadata": metadata or {}
        }
        
        # Write to log file
        with self.lock:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Track for suspicious activity
        if self.enable_suspicious_tracking:
            self._track_suspicious_activity(username, success, ip_address, log_entry)
    
    def _track_suspicious_activity(self, username: str, success: bool, 
                                   ip_address: Optional[str], log_entry: Dict):
        """Track patterns that may indicate suspicious activity"""
        current_time = time.time()
        
        # Track failed attempts per user
        if not success:
            self.failed_attempts[username].append(current_time)
            
            # Clean old attempts (older than 1 hour)
            self.failed_attempts[username] = [
                t for t in self.failed_attempts[username] 
                if current_time - t < 3600
            ]
            
            # Check for rapid failed attempts
            if len(self.failed_attempts[username]) >= 3:
                recent_failures = [
                    t for t in self.failed_attempts[username]
                    if current_time - t < 300  # Last 5 minutes
                ]
                if len(recent_failures) >= 3:
                    self._flag_suspicious("rapid_failed_attempts", username, {
                        "count": len(recent_failures),
                        "window": "5_minutes"
                    })
        
        # Track IP-based patterns
        if ip_address:
            self.ip_attempts[ip_address].append({
                'username': username,
                'timestamp': current_time,
                'success': success
            })
            
            # Clean old attempts
            self.ip_attempts[ip_address] = [
                a for a in self.ip_attempts[ip_address]
                if current_time - a['timestamp'] < 3600
            ]
            
            # Check for multiple users from same IP
            recent_ips = [
                a for a in self.ip_attempts[ip_address]
                if current_time - a['timestamp'] < 1800  # Last 30 minutes
            ]
            unique_users = set(a['username'] for a in recent_ips)
            
            if len(unique_users) >= 5:
                self._flag_suspicious("multiple_users_same_ip", ip_address, {
                    "unique_users": len(unique_users),
                    "attempts": len(recent_ips)
                })
        
        # Check for specific failure patterns
        if not success and log_entry.get('reason'):
            reason = log_entry['reason']
            
            # Track consecutive same-reason failures
            pattern_key = f"{username}:{reason}"
            self.suspicious_patterns[pattern_key] += 1
            
            if self.suspicious_patterns[pattern_key] >= 5:
                self._flag_suspicious("repeated_same_failure", username, {
                    "reason": reason,
                    "count": self.suspicious_patterns[pattern_key]
                })
    
    def _flag_suspicious(self, pattern_type: str, identifier: str, details: Dict):
        """Flag and log suspicious activity"""
        suspicious_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "SUSPICIOUS_ACTIVITY",
            "pattern": pattern_type,
            "identifier": identifier,
            "details": details
        }
        
        # Write to separate suspicious activity log
        suspicious_log_path = self.log_path.replace('.log', '_suspicious.log')
        with self.lock:
            with open(suspicious_log_path, 'a') as f:
                f.write(json.dumps(suspicious_entry) + '\n')
    
    def get_user_history(self, username: str, hours: int = 24) -> List[Dict]:
        """Get verification history for a specific user"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = []
        
        with self.lock:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            if entry['username'] == username and entry_time >= cutoff_time:
                                history.append(entry)
                        except Exception:
                            continue
        
        return history
    
    def get_suspicious_activity(self, hours: int = 24) -> List[Dict]:
        """Get all suspicious activity from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        suspicious = []
        
        suspicious_log_path = self.log_path.replace('.log', '_suspicious.log')
        
        with self.lock:
            if os.path.exists(suspicious_log_path):
                with open(suspicious_log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            if entry_time >= cutoff_time:
                                suspicious.append(entry)
                        except Exception:
                            continue
        
        return suspicious
    
    def get_statistics(self, hours: int = 24) -> Dict:
        """Get verification statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0,
            'unique_users': set(),
            'failure_reasons': defaultdict(int),
            'avg_liveness_score': [],
            'avg_quality_score': [],
            'time_period_hours': hours
        }
        
        with self.lock:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            if entry_time >= cutoff_time:
                                stats['total_attempts'] += 1
                                stats['unique_users'].add(entry['username'])
                                
                                if entry['success']:
                                    stats['successful'] += 1
                                else:
                                    stats['failed'] += 1
                                    if entry.get('reason'):
                                        stats['failure_reasons'][entry['reason']] += 1
                                
                                if entry.get('liveness_score') is not None:
                                    stats['avg_liveness_score'].append(entry['liveness_score'])
                                if entry.get('quality_score') is not None:
                                    stats['avg_quality_score'].append(entry['quality_score'])
                        except Exception:
                            continue
        
        # Calculate averages
        stats['unique_users'] = len(stats['unique_users'])
        if stats['avg_liveness_score']:
            stats['avg_liveness_score'] = sum(stats['avg_liveness_score']) / len(stats['avg_liveness_score'])
        else:
            stats['avg_liveness_score'] = 0.0
        
        if stats['avg_quality_score']:
            stats['avg_quality_score'] = sum(stats['avg_quality_score']) / len(stats['avg_quality_score'])
        else:
            stats['avg_quality_score'] = 0.0
        
        stats['failure_reasons'] = dict(stats['failure_reasons'])
        
        return stats
    
    def cleanup_old_logs(self):
        """Remove log entries older than retention period"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        # Read all entries
        valid_entries = []
        with self.lock:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            if entry_time >= cutoff_time:
                                valid_entries.append(line.strip())
                        except Exception:
                            continue
                
                # Rewrite log file with only valid entries
                with open(self.log_path, 'w') as f:
                    for entry in valid_entries:
                        f.write(entry + '\n')
        
        # Also cleanup suspicious log
        suspicious_log_path = self.log_path.replace('.log', '_suspicious.log')
        valid_suspicious = []
        
        with self.lock:
            if os.path.exists(suspicious_log_path):
                with open(suspicious_log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            if entry_time >= cutoff_time:
                                valid_suspicious.append(line.strip())
                        except Exception:
                            continue
                
                with open(suspicious_log_path, 'w') as f:
                    for entry in valid_suspicious:
                        f.write(entry + '\n')
    
    def is_user_suspicious(self, username: str) -> bool:
        """Check if a user has suspicious activity patterns"""
        current_time = time.time()
        
        # Check failed attempts
        if username in self.failed_attempts:
            recent_failures = [
                t for t in self.failed_attempts[username]
                if current_time - t < 1800  # Last 30 minutes
            ]
            if len(recent_failures) >= 5:
                return True
        
        return False
    
    def get_risk_score(self, username: str, ip_address: Optional[str] = None) -> float:
        """
        Calculate risk score for a verification attempt (0-1, higher = riskier)
        Used for adaptive threshold adjustment
        """
        risk_score = 0.0
        current_time = time.time()
        
        # Factor 1: Recent failed attempts (0-0.4)
        if username in self.failed_attempts:
            recent_failures = [
                t for t in self.failed_attempts[username]
                if current_time - t < 1800  # Last 30 minutes
            ]
            risk_score += min(len(recent_failures) * 0.08, 0.4)
        
        # Factor 2: IP reputation (0-0.3)
        if ip_address and ip_address in self.ip_attempts:
            recent_ip_attempts = [
                a for a in self.ip_attempts[ip_address]
                if current_time - a['timestamp'] < 1800
            ]
            failed_from_ip = sum(1 for a in recent_ip_attempts if not a['success'])
            if len(recent_ip_attempts) > 0:
                fail_rate = failed_from_ip / len(recent_ip_attempts)
                risk_score += fail_rate * 0.3
        
        # Factor 3: Pattern matching (0-0.3)
        for pattern_key, count in self.suspicious_patterns.items():
            if username in pattern_key and count >= 3:
                risk_score += 0.15
                break
        
        return min(risk_score, 1.0)


# Global logger instance
_logger_instance = None

def get_logger(config: Optional[Dict] = None) -> VerificationLogger:
    """Get or create global logger instance"""
    global _logger_instance
    
    if _logger_instance is None:
        if config:
            _logger_instance = VerificationLogger(
                log_path=config.get('logging', {}).get('log_path', 'logs/verification.log'),
                retention_days=config.get('logging', {}).get('retention_days', 30),
                enable_suspicious_tracking=config.get('logging', {}).get('log_suspicious_activity', True)
            )
        else:
            _logger_instance = VerificationLogger()
    
    return _logger_instance


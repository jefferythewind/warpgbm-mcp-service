"""
Feedback storage adapter - works seamlessly in Modal and local environments
"""

import os
import pandas as pd
from typing import Dict, Any


class FeedbackStorage:
    """Adapter for storing feedback in parquet files across different environments"""
    
    def __init__(self):
        # Auto-detect environment
        self.is_modal = os.path.exists("/data")
        self.feedback_dir = "/data" if self.is_modal else "data"
        self.feedback_file = os.path.join(self.feedback_dir, "feedback.parquet")
        
        # Ensure directory exists
        os.makedirs(self.feedback_dir, exist_ok=True)
        
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Save feedback to parquet file.
        
        Args:
            feedback_data: Dictionary with feedback fields (id, type, message, etc.)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create DataFrame from feedback
            new_feedback_df = pd.DataFrame([feedback_data])
            
            # Append to existing file or create new one
            if os.path.exists(self.feedback_file):
                existing_df = pd.read_parquet(self.feedback_file)
                combined_df = pd.concat([existing_df, new_feedback_df], ignore_index=True)
                combined_df.to_parquet(self.feedback_file, index=False)
            else:
                new_feedback_df.to_parquet(self.feedback_file, index=False)
            
            # Modal volumes auto-commit on function exit, no action needed
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to save feedback: {str(e)}")
            return False
    
    def get_all_feedback(self) -> pd.DataFrame:
        """
        Retrieve all feedback from storage.
        
        Returns:
            DataFrame with all feedback, or empty DataFrame if file doesn't exist
        """
        try:
            if os.path.exists(self.feedback_file):
                return pd.read_parquet(self.feedback_file)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Failed to read feedback: {str(e)}")
            return pd.DataFrame()
    
    def get_feedback_count(self) -> int:
        """Get total number of feedback entries"""
        try:
            if os.path.exists(self.feedback_file):
                df = pd.read_parquet(self.feedback_file)
                return len(df)
            return 0
        except:
            return 0
    
    @property
    def environment(self) -> str:
        """Return current environment name"""
        return "modal" if self.is_modal else "local"


# Global singleton instance
_feedback_storage = None


def get_feedback_storage() -> FeedbackStorage:
    """Get or create the global feedback storage instance"""
    global _feedback_storage
    if _feedback_storage is None:
        _feedback_storage = FeedbackStorage()
    return _feedback_storage




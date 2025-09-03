"""Smart date parsing for journal entries."""

import re
from datetime import datetime, timedelta
from typing import Optional


class DateParser:
    """Handles flexible date parsing for journal entries."""
    
    @staticmethod
    def parse_date(date_input: str) -> str:
        """
        Parse various date formats into YYYY-MM-DD format.
        
        Supported formats:
        - today, now
        - yesterday 
        - monday, tuesday, etc. (last occurrence)
        - 3 days ago, 2 weeks ago
        - 2025-01-15, 2025/01/15, 01/15/2025
        - 01-15, 15 (current year assumed)
        """
        date_input = date_input.lower().strip()
        today = datetime.now()
        
        # Handle "today" and "now"
        if date_input in ["today", "now", ""]:
            return today.strftime("%Y-%m-%d")
        
        # Handle "yesterday"
        if date_input == "yesterday":
            return (today - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Handle weekdays (last occurrence) and "last [weekday]"
        weekdays = {
            "monday": 0, "mon": 0,
            "tuesday": 1, "tue": 1, "tues": 1,
            "wednesday": 2, "wed": 2,
            "thursday": 3, "thu": 3, "thurs": 3,
            "friday": 4, "fri": 4,
            "saturday": 5, "sat": 5,
            "sunday": 6, "sun": 6
        }
        
        # Check for "last [weekday]" pattern
        last_weekday_pattern = r"^last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun|tues|thurs)$"
        match = re.match(last_weekday_pattern, date_input)
        if match:
            weekday_name = match.group(1)
            target_weekday = weekdays[weekday_name]
            current_weekday = today.weekday()
            
            # Calculate days back to last occurrence
            if target_weekday <= current_weekday:
                days_back = current_weekday - target_weekday
            else:
                days_back = 7 - (target_weekday - current_weekday)
            
            # If it's the same day, assume they mean last week
            if days_back == 0:
                days_back = 7
                
            target_date = today - timedelta(days=days_back)
            return target_date.strftime("%Y-%m-%d")
        
        # Handle plain weekdays (last occurrence)
        if date_input in weekdays:
            target_weekday = weekdays[date_input]
            current_weekday = today.weekday()
            
            # Calculate days back to last occurrence
            if target_weekday <= current_weekday:
                days_back = current_weekday - target_weekday
            else:
                days_back = 7 - (target_weekday - current_weekday)
            
            # If it's the same day, assume they mean last week
            if days_back == 0:
                days_back = 7
                
            target_date = today - timedelta(days=days_back)
            return target_date.strftime("%Y-%m-%d")
        
        # Handle "this [weekday]" pattern (last occurrence in current week)
        this_weekday_pattern = r"^this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun|tues|thurs)$"
        match = re.match(this_weekday_pattern, date_input)
        if match:
            weekday_name = match.group(1)
            target_weekday = weekdays[weekday_name]
            current_weekday = today.weekday()
            
            # Calculate days back to this week's occurrence
            if target_weekday <= current_weekday:
                days_back = current_weekday - target_weekday
            else:
                # Future day this week, go to last week's occurrence
                days_back = 7 - (target_weekday - current_weekday)
                
            target_date = today - timedelta(days=days_back)
            return target_date.strftime("%Y-%m-%d")

        # Handle relative dates like "3 days ago", "2 weeks ago", "two weeks ago"
        # First try numeric pattern
        relative_pattern = r"(\d+)\s+(day|days|week|weeks)\s+ago"
        match = re.match(relative_pattern, date_input)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            
            if unit.startswith("day"):
                target_date = today - timedelta(days=number)
            elif unit.startswith("week"):
                target_date = today - timedelta(weeks=number)
            
            return target_date.strftime("%Y-%m-%d")
        
        # Handle word-based numbers like "two weeks ago", "three days ago"
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a": 1, "an": 1, "couple": 2, "few": 3
        }
        
        word_relative_pattern = r"(one|two|three|four|five|six|seven|eight|nine|ten|a|an|couple|few)\s+(day|days|week|weeks)\s+ago"
        match = re.match(word_relative_pattern, date_input)
        if match:
            number_word = match.group(1)
            unit = match.group(2)
            number = number_words[number_word]
            
            if unit.startswith("day"):
                target_date = today - timedelta(days=number)
            elif unit.startswith("week"):
                target_date = today - timedelta(weeks=number)
            
            return target_date.strftime("%Y-%m-%d")
        
        # Handle explicit dates
        # Try YYYY-MM-DD format first
        if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", date_input):
            try:
                parsed = datetime.strptime(date_input, "%Y-%m-%d")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        # Try YYYY/MM/DD format
        if re.match(r"^\d{4}/\d{1,2}/\d{1,2}$", date_input):
            try:
                parsed = datetime.strptime(date_input, "%Y/%m/%d")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        # Try MM/DD/YYYY format
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", date_input):
            try:
                parsed = datetime.strptime(date_input, "%m/%d/%Y")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        # Try MM-DD format (current year assumed)
        if re.match(r"^\d{1,2}-\d{1,2}$", date_input):
            try:
                month_day = date_input.split("-")
                parsed = datetime(today.year, int(month_day[0]), int(month_day[1]))
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        # Try just day number (current month/year assumed)
        if re.match(r"^\d{1,2}$", date_input):
            try:
                day = int(date_input)
                parsed = datetime(today.year, today.month, day)
                
                # If the date is in the future, assume they meant last month
                if parsed > today:
                    if today.month == 1:
                        parsed = datetime(today.year - 1, 12, day)
                    else:
                        parsed = datetime(today.year, today.month - 1, day)
                
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        # If nothing matches, raise an exception
        raise ValueError(f"Unable to parse date: '{date_input}'. Please try formats like 'today', 'yesterday', 'last monday', '2025-08-30', or '3 days ago'.")
    
    @staticmethod
    def format_date_display(date_str: str) -> str:
        """Format a date string for display."""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now()
            
            if date_obj.date() == today.date():
                return "today"
            elif date_obj.date() == (today - timedelta(days=1)).date():
                return "yesterday"
            else:
                return date_obj.strftime("%A, %B %d, %Y")
        except ValueError:
            return date_str
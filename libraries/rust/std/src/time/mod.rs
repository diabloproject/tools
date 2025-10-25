use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct DateTime(i64);

impl DateTime {
    pub fn now() -> Self {
        let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        DateTime(since_epoch.as_millis() as i64)
    }

    pub fn from_millis(millis: i64) -> Self {
        DateTime(millis)
    }

    pub fn from_components(
        year: i64,
        month: i64,
        day: i64,
        hours: i64,
        minutes: i64,
        seconds: i64,
        millis: i64,
    ) -> Result<Self, DateParseError> {
        // Basic validation
        if !(1..=12).contains(&month)
            || !(1..=31).contains(&day)
            || hours > 23
            || minutes > 59
            || seconds > 59
            || millis > 999
        {
            return Err(DateParseError);
        }

        let days_in_month = match month {
            4 | 6 | 9 | 11 => 30,
            2 => {
                if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                    29
                } else {
                    28
                }
            }
            _ => 31,
        };

        if day > days_in_month {
            return Err(DateParseError);
        }

        // Calculate total days from year 1970
        let mut days = (year - 1970) * 365;

        // Add leap days between 1970 and target year
        let leap_adjust = if year > 1970 {
            let y1 = 1969;
            let y2 = year - 1;
            (y2 / 4 - y1 / 4) - (y2 / 100 - y1 / 100) + (y2 / 400 - y1 / 400)
        } else if year < 1970 {
            let y1 = year;
            let y2 = 1969;
            -((y2 / 4 - y1 / 4) - (y2 / 100 - y1 / 100) + (y2 / 400 - y1 / 400))
        } else {
            0
        };
        days += leap_adjust;

        let days_before_month = match month {
            2 => 31,
            3 => 59,
            4 => 90,
            5 => 120,
            6 => 151,
            7 => 181,
            8 => 212,
            9 => 243,
            10 => 273,
            11 => 304,
            12 => 334,
            _ => 0,
        };

        days += days_before_month + day - 1;

        let total_millis = days * 24 * 60 * 60 * 1000
            + hours * 60 * 60 * 1000
            + minutes * 60 * 1000
            + seconds * 1000
            + millis;
        Ok(DateTime(total_millis))
    }
    pub fn to_millis(&self) -> i64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DateParseError;

impl std::fmt::Display for DateParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid date")
    }
}

impl std::error::Error for DateParseError {}

impl FromStr for DateTime {
    type Err = DateParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Format: "2024-01-01T00:00:00.000Z"
        if !s.ends_with('Z') {
            return Err(DateParseError);
        }

        let parts: Vec<&str> = s[..s.len() - 1].split('T').collect();
        if parts.len() != 2 {
            return Err(DateParseError);
        }

        let date_parts: Vec<&str> = parts[0].split('-').collect();
        let time_parts: Vec<&str> = parts[1].split(':').collect();

        if date_parts.len() != 3 || time_parts.len() != 3 {
            return Err(DateParseError);
        }

        let year: i64 = date_parts[0].parse().map_err(|_| DateParseError)?;
        let month: i64 = date_parts[1].parse().map_err(|_| DateParseError)?;
        let day: i64 = date_parts[2].parse().map_err(|_| DateParseError)?;

        let time_sec_parts: Vec<&str> = time_parts[2].split('.').collect();
        if time_sec_parts.len() != 2 {
            return Err(DateParseError);
        }

        let hours: i64 = time_parts[0].parse().map_err(|_| DateParseError)?;
        let minutes: i64 = time_parts[1].parse().map_err(|_| DateParseError)?;
        let seconds: i64 = time_sec_parts[0].parse().map_err(|_| DateParseError)?;
        let millis: i64 = time_sec_parts[1].parse().map_err(|_| DateParseError)?;

        DateTime::from_components(year, month, day, hours, minutes, seconds, millis)
    }
}

impl std::fmt::Debug for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DateTime({:?})", self.0)
    }
}

impl std::fmt::Display for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_millis = self.0;

        // Calculate total days and remaining milliseconds in day
        let total_days = if total_millis >= 0 {
            total_millis / (24 * 60 * 60 * 1000)
        } else {
            (total_millis - (24 * 60 * 60 * 1000 - 1)) / (24 * 60 * 60 * 1000)
        };

        let mut millis_in_day = total_millis - total_days * 24 * 60 * 60 * 1000;
        if millis_in_day < 0 {
            millis_in_day += 24 * 60 * 60 * 1000;
        }

        // Calculate year from total days
        let mut year = 1970;
        let mut days_remaining = total_days;

        if total_days >= 0 {
            loop {
                let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                    366
                } else {
                    365
                };
                if days_remaining < days_in_year {
                    break;
                }
                days_remaining -= days_in_year;
                year += 1;
            }
        } else {
            loop {
                year -= 1;
                let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                    366
                } else {
                    365
                };
                days_remaining += days_in_year;
                if days_remaining >= 0 {
                    break;
                }
            }
        }

        // Calculate month from remaining days
        let month_days = [
            31,
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                29
            } else {
                28
            },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        let mut month = 1;
        for days in month_days.iter() {
            if days_remaining < *days {
                break;
            }
            days_remaining -= *days;
            month += 1;
        }

        let day = days_remaining + 1;

        // Calculate time components from milliseconds in day
        let hours = millis_in_day / (60 * 60 * 1000);
        millis_in_day %= 60 * 60 * 1000;

        let minutes = millis_in_day / (60 * 1000);
        millis_in_day %= 60 * 1000;

        let seconds = millis_in_day / 1000;
        let millis = millis_in_day % 1000;

        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            year, month, day, hours, minutes, seconds, millis
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_after_epoch() {
        let date = "2024-01-01T00:00:00.000Z".parse::<DateTime>().unwrap();
        assert_eq!(date.to_string(), "2024-01-01T00:00:00.000Z");
    }

    #[test]
    fn test_parse_before_epoch() {
        let date = "1969-12-31T23:59:59.999Z".parse::<DateTime>().unwrap();
        assert_eq!(date.to_string(), "1969-12-31T23:59:59.999Z");
    }

    #[test]
    fn test_now() {
        let now = DateTime::now();
        let _parsed = now.to_string().parse::<DateTime>().unwrap();
    }

    #[test]
    fn test_invalid_formats() {
        assert!("2024-01-01".parse::<DateTime>().is_err());
        assert!("2024-01-01T00:00:00.000".parse::<DateTime>().is_err());
        assert!("2024-13-01T00:00:00.000Z".parse::<DateTime>().is_err());
        assert!("2024-01-32T00:00:00.000Z".parse::<DateTime>().is_err());
        assert!("2024-01-01T24:00:00.000Z".parse::<DateTime>().is_err());
        assert!("2024-01-01T00:60:00.000Z".parse::<DateTime>().is_err());
        assert!("2024-01-01T00:00:60.000Z".parse::<DateTime>().is_err());
    }
}

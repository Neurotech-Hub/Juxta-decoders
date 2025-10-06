#!/usr/bin/env python3
"""
Generate simulated social interaction data for 7-day deployment
Creates realistic patterns with 9 neighboring devices, circadian rhythms, and social behaviors
"""

import csv
import json
import random
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

def generate_random_mac_suffix():
    """Generate a random 6-character MAC address suffix (hexadecimal)"""
    chars = 'ABCDEF0123456789'  # Hexadecimal characters only
    return ''.join(random.choice(chars) for _ in range(6))

def generate_simulation_data():
    """Generate 7 days of simulated social interaction data"""
    
    # Device configuration
    source_device = "6C38B0"
    neighboring_devices = [
        generate_random_mac_suffix() for _ in range(9)
    ]
    
    # Social groups (devices that travel together)
    # 2-3 consistently detected devices, others sporadic
    social_groups = {
        "core_group": neighboring_devices[:3],  # First 3 devices - consistently detected
        "occasional": neighboring_devices[3:5],  # Next 2 devices - occasionally detected
        "rare": neighboring_devices[5:]  # Remaining devices - rarely detected
    }
    
    # Start date (4 days ago)
    start_date = datetime.now() - timedelta(days=4)
    start_date = start_date.replace(hour=6, minute=0, second=0, microsecond=0)
    
    # Generate data for 4 days
    records = []
    current_time = start_date
    
    # Add boot event at the beginning
    boot_record = create_record(
        current_time, source_device, "system_event", "boot", 
        0, 0, 100, 25, [], []
    )
    records.append(boot_record)
    current_time += timedelta(minutes=1)
    
    # Generate 4 days of data (every minute)
    for day in range(4):
        current_date = start_date + timedelta(days=day)
        
        # Generate data for each hour of the day
        for hour in range(24):
            current_time = current_date.replace(hour=hour, minute=0)
            
            # Generate 60 minutes of data for this hour
            for minute in range(60):
                if current_time >= datetime.now():
                    break
                    
                # Calculate circadian factors
                circadian_factor = calculate_circadian_factor(hour)
                activity_level = calculate_activity_level(hour, day)
                
                # Determine record type based on activity and social patterns
                record_type, detected_devices = determine_record_type(
                    hour, day, social_groups, neighboring_devices, activity_level
                )
                
                # Generate motion count (higher during day, lower at night)
                motion_count = generate_motion_count(hour, activity_level)
                
                # Generate battery level (smooth linear decline over 4 days)
                battery_level = generate_battery_level(day, hour, minute, motion_count)
                
                # Generate temperature (slight increase when motion is low)
                temperature = generate_temperature(hour, motion_count)
                
                # Generate device data if this is a device scan
                device_macs = []
                device_rssis = []
                if record_type == "device_scan" and detected_devices:
                    device_macs, device_rssis = generate_device_data(
                        detected_devices, social_groups, hour
                    )
                
                # Create the record
                record = create_record(
                    current_time, source_device, record_type, "",
                    len(detected_devices), motion_count, battery_level, temperature,
                    device_macs, device_rssis
                )
                records.append(record)
                
                current_time += timedelta(minutes=1)
    
    return records

def calculate_circadian_factor(hour: int) -> float:
    """Calculate circadian rhythm factor (0-1) with realistic variation"""
    # Base circadian pattern
    if 6 <= hour <= 22:
        base_factor = 0.7 + 0.3 * math.sin((hour - 6) * math.pi / 16)
    else:
        base_factor = 0.1 + 0.2 * math.sin((hour + 6) * math.pi / 12)
    
    # Add significant random variation (±30%)
    variation = random.uniform(-0.3, 0.3)
    return max(0.0, min(1.0, base_factor + variation))

def calculate_activity_level(hour: int, day: int) -> float:
    """Calculate overall activity level (0-1) with realistic variation"""
    circadian = calculate_circadian_factor(hour)
    
    # Add day-to-day variation (some days are more active than others)
    day_variation = random.uniform(0.3, 1.2)
    
    # Add random "events" that can spike or drop activity
    event_factor = 1.0
    if random.random() < 0.05:  # 5% chance of unusual activity
        event_factor = random.uniform(0.2, 2.0)
    
    return max(0.0, min(1.0, circadian * day_variation * event_factor))

def determine_record_type(hour: int, day: int, social_groups: Dict, 
                         neighboring_devices: List[str], activity_level: float) -> tuple:
    """Determine record type and detected devices with realistic variation"""
    
    # Base probability of device detection
    base_detection_prob = 0.4 + 0.3 * activity_level
    
    # Add time-based variation
    if 6 <= hour <= 22:
        time_factor = random.uniform(0.8, 1.5)
    else:
        time_factor = random.uniform(0.2, 0.8)
    
    detection_prob = base_detection_prob * time_factor
    
    # Add random "social events" - periods of high/low social activity
    if random.random() < 0.1:  # 10% chance of social event
        if random.random() < 0.5:
            detection_prob *= random.uniform(1.5, 2.5)  # High social activity
        else:
            detection_prob *= random.uniform(0.2, 0.5)   # Low social activity
    
    # Determine which devices to detect
    detected_devices = []
    
    if random.random() < detection_prob:
        # Core group devices (2-3 consistently detected)
        core_devices = social_groups["core_group"]
        num_core = random.randint(1, min(3, len(core_devices)))
        detected_devices = random.sample(core_devices, num_core)
        
        # Occasionally add other devices
        if random.random() < 0.3:  # 30% chance to add occasional devices
            occasional_devices = social_groups["occasional"]
            if occasional_devices:
                num_occasional = random.randint(1, min(2, len(occasional_devices)))
                additional = random.sample(occasional_devices, num_occasional)
                detected_devices.extend(additional)
        
        # Rarely add rare devices
        if random.random() < 0.1:  # 10% chance to add rare devices
            rare_devices = social_groups["rare"]
            if rare_devices:
                num_rare = random.randint(1, min(2, len(rare_devices)))
                additional = random.sample(rare_devices, num_rare)
                detected_devices.extend(additional)
    
    # Determine record type
    if detected_devices:
        return "device_scan", detected_devices
    else:
        return "no_device_proximity", []

def generate_motion_count(hour: int, activity_level: float) -> int:
    """Generate motion count with realistic variation (max 60 per minute)"""
    # More realistic base motion (0-30 range)
    base_motion = random.uniform(0, 30) + 20 * activity_level
    
    # Add time-based variation (but much more random)
    if 6 <= hour <= 22:
        time_factor = random.uniform(0.5, 2.0)  # Reduced max variation
    else:
        time_factor = random.uniform(0.1, 0.8)  # Some night activity
    
    # Add random "bursts" of activity
    if random.random() < 0.05:  # 5% chance of activity burst
        burst_factor = random.uniform(1.5, 3.0)  # Reduced burst intensity
    else:
        burst_factor = 1.0
    
    # Add random "quiet periods"
    if random.random() < 0.1:  # 10% chance of quiet period
        quiet_factor = random.uniform(0.1, 0.5)
    else:
        quiet_factor = 1.0
    
    motion = int(base_motion * time_factor * burst_factor * quiet_factor)
    return max(0, min(60, motion))  # Cap at 60 per minute

def generate_battery_level(day: int, hour: int, minute: int, motion_count: int) -> int:
    """Generate battery level with realistic coin cell drain (100% to 50%)"""
    # Calculate total minutes elapsed since start
    total_minutes = day * 24 * 60 + hour * 60 + minute
    
    # Realistic coin cell drain: 100% to 50% over 4 days
    # Use exponential-like decay that's more realistic for coin cells
    decay_factor = total_minutes / 5760  # 0 to 1 over 4 days
    base_battery = 100 - (decay_factor ** 1.2) * 50  # Exponential decay to 50%
    
    # Add variation based on activity (more activity = more battery drain)
    # Scale motion drain based on total time elapsed
    activity_drain = motion_count * random.uniform(0.0005, 0.002) * (1 + decay_factor)
    
    # Add random battery "events" (charging, power saving, etc.)
    if random.random() < 0.005:  # 0.5% chance of battery event (very rare)
        if random.random() < 0.1:  # 10% chance of charging (very rare)
            battery_boost = random.uniform(1, 3)
        else:  # 90% chance of power saving
            battery_boost = random.uniform(-1, -3)
    else:
        battery_boost = 0
    
    # Add realistic random variation (±0.5%)
    variation = random.uniform(-0.5, 0.5)
    
    battery = int(base_battery - activity_drain + battery_boost + variation)
    return max(50, min(100, battery))  # Keep between 50-100%

def generate_temperature(hour: int, motion_count: int) -> int:
    """Generate temperature with realistic variation (23-32°C)"""
    # More realistic base temperature range
    base_temp = random.uniform(25, 30)
    
    # Daily cycle (but more random)
    daily_cycle = random.uniform(-2, 2) * math.sin((hour - 6) * math.pi / 12)
    
    # Motion effect (more motion = higher temp, but not linear)
    motion_effect = motion_count * random.uniform(0.01, 0.05)
    
    # Add random temperature "events" (weather, environment changes)
    if random.random() < 0.05:  # 5% chance of temperature event
        temp_event = random.uniform(-2, 2)  # Reduced event intensity
    else:
        temp_event = 0
    
    # Add random variation
    variation = random.uniform(-1, 1)
    
    temp = int(base_temp + daily_cycle + motion_effect + temp_event + variation)
    return max(23, min(32, temp))  # Keep between 23-32°C

def generate_device_data(detected_devices: List[str], social_groups: Dict, 
                        hour: int) -> tuple:
    """Generate device MACs and RSSI values with realistic variation and jitter"""
    device_macs = []
    device_rssis = []
    
    for device in detected_devices:
        device_macs.append(device)
        
        # Generate RSSI based on social group strength with more realistic variation
        if device in social_groups["core_group"]:
            # Core group - consistently closer but with more variation
            base_rssi = random.uniform(-75, -58)  # Wider range, can go below -60
        elif device in social_groups["occasional"]:
            # Occasional group - medium range with variation
            base_rssi = random.uniform(-85, -65)  # Wider range
        else:
            # Rare group - farther range with variation
            base_rssi = random.uniform(-95, -75)  # Wider range
        
        # Add significant random variation (±8 dBm)
        rssi_variation = random.uniform(-8, 8)
        
        # Add time-based variation (signal strength can vary significantly)
        if 6 <= hour <= 22:
            time_variation = random.uniform(-5, 5)
        else:
            time_variation = random.uniform(-8, 2)  # Generally weaker at night
        
        # Add random "signal events" (obstacles, interference, etc.)
        if random.random() < 0.08:  # 8% chance of signal event
            signal_event = random.uniform(-12, 6)  # More dramatic events
        else:
            signal_event = 0
        
        # Add device-specific jitter (some devices have more stable signals)
        device_jitter = random.uniform(-3, 3)
        
        # Add random "environmental" jitter (weather, obstacles, etc.)
        env_jitter = random.uniform(-4, 4)
        
        rssi = int(base_rssi + rssi_variation + time_variation + signal_event + device_jitter + env_jitter)
        # Remove artificial upper limit - let RSSI vary naturally
        # Only apply reasonable bounds to prevent extreme outliers
        rssi = max(-110, min(-40, rssi))  # Very wide range for natural variation
        device_rssis.append(rssi)
    
    return device_macs, device_rssis

def create_record(timestamp: datetime, source_device: str, record_type: str, 
                 event_name: str, device_count: int, motion_count: int, 
                 battery_level: int, temperature: int, device_macs: List[str], 
                 device_rssis: List[int]) -> Dict[str, Any]:
    """Create a single record"""
    
    # Convert to minute of day
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    
    # Create UTC timestamp (assuming device records in UTC)
    utc_timestamp = timestamp.timestamp()
    local_timestamp = utc_timestamp  # For simulation, same as UTC
    
    # Format datetimes
    utc_datetime = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
    local_datetime = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        'minute_of_day': minute_of_day,
        'utc_timestamp': utc_timestamp,
        'local_timestamp': local_timestamp,
        'utc_datetime': utc_datetime,
        'local_datetime': local_datetime,
        'record_type': record_type,
        'event_name': event_name,
        'device_count': device_count,
        'motion_count': motion_count,
        'battery_level': battery_level,
        'temperature_c': temperature,
        'device_macs': json.dumps(device_macs) if device_macs else '',
        'device_rssis': json.dumps(device_rssis) if device_rssis else '',
        'source_file': f'JX_{source_device}_250929.txt',
        'source_device_id': source_device,
        'source_recording_date': timestamp.strftime('%Y-%m-%d')
    }

def main():
    """Generate and save simulation data"""
    print("Generating 4-day social interaction simulation...")
    print("Source device: 6C38B0")
    print("Neighboring devices: 9 devices with realistic social group patterns")
    print("Features: High variation, random events, realistic animal behavior")
    print("MAC addresses: Randomly generated 6-character codes (A-F, 0-9)")
    print()
    
    # Generate data
    records = generate_simulation_data()
    
    # Save to CSV
    output_file = "./analysis_social/simulation/6C38B0_250929_events.csv"
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'minute_of_day', 'utc_timestamp', 'local_timestamp', 'utc_datetime', 'local_datetime',
            'record_type', 'event_name', 'device_count', 'motion_count', 'battery_level', 'temperature_c',
            'device_macs', 'device_rssis', 'source_file', 'source_device_id', 'source_recording_date'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in records:
            writer.writerow(record)
    
    print(f"Generated {len(records)} records")
    print(f"Saved to: {output_file}")
    
    # Print summary
    record_types = {}
    total_motion = 0
    total_devices = 0
    
    for record in records:
        rt = record['record_type']
        record_types[rt] = record_types.get(rt, 0) + 1
        total_motion += record['motion_count']
        total_devices += record['device_count']
    
    print(f"\nSummary:")
    print(f"  Record types: {record_types}")
    print(f"  Total motion events: {total_motion}")
    print(f"  Total device detections: {total_devices}")
    print(f"  Time span: {records[0]['local_datetime']} to {records[-1]['local_datetime']}")

if __name__ == "__main__":
    main()

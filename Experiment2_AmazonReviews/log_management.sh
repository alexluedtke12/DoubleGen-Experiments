#!/bin/bash

# Log Management Functions for CF Reviews Pipeline
# ================================================
# This script provides reusable functions for managing logs across all pipeline steps
# Source this file in each step script to use these functions

# Function to clean all log files in a specified directory
# Usage: clean_logs "/path/to/logs/directory"
clean_logs() {
    local log_dir="$1"
    
    if [ -z "$log_dir" ]; then
        echo "[ERROR] Log directory not specified"
        return 1
    fi
    
    if [ ! -d "$log_dir" ]; then
        echo "[WARNING] Log directory does not exist: $log_dir"
        return 1
    fi
    
    # Count existing log files
    local log_count=$(find "$log_dir" -name "*.log" -type f 2>/dev/null | wc -l)
    
    if [ "$log_count" -gt 0 ]; then
        echo "[INFO] Cleaning $log_count log files from $log_dir"
        rm -f "$log_dir"/*.log
        echo "[INFO] Log files cleaned successfully"
    else
        echo "[INFO] No log files to clean in $log_dir"
    fi
}

# Function to create a new log file with start timestamp
# Usage: start_log "/path/to/logfile.log" "Job Description"
start_log() {
    local log_file="$1"
    local job_description="$2"
    
    if [ -z "$log_file" ]; then
        echo "[ERROR] Log file path not specified"
        return 1
    fi
    
    # Ensure log directory exists
    local log_dir=$(dirname "$log_file")
    mkdir -p "$log_dir"
    
    # Write header with start timestamp
    {
        echo "========================================"
        echo "$job_description"
        echo "========================================"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "Hostname: $(hostname)"
        echo "Working Directory: $(pwd)"
        echo "========================================"
        echo ""
    } > "$log_file"
    
    echo "[INFO] Log started at: $log_file"
}

# Function to append end timestamp to log file
# Usage: end_log "/path/to/logfile.log" $exit_code
end_log() {
    local log_file="$1"
    local exit_code="${2:-0}"
    
    if [ -z "$log_file" ]; then
        echo "[ERROR] Log file path not specified"
        return 1
    fi
    
    if [ ! -f "$log_file" ]; then
        echo "[WARNING] Log file does not exist: $log_file"
        return 1
    fi
    
    # Append footer with end timestamp
    {
        echo ""
        echo "========================================"
        if [ "$exit_code" -eq 0 ]; then
            echo "Job Status: COMPLETED SUCCESSFULLY"
        else
            echo "Job Status: FAILED (Exit Code: $exit_code)"
        fi
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
        
        # Calculate runtime if possible
        local start_time=$(grep "Start Time:" "$log_file" | head -1 | cut -d':' -f2- | xargs)
        if [ -n "$start_time" ]; then
            local start_epoch=$(date -d "$start_time" +%s 2>/dev/null)
            local end_epoch=$(date +%s)
            if [ -n "$start_epoch" ]; then
                local runtime=$((end_epoch - start_epoch))
                local hours=$((runtime / 3600))
                local minutes=$(((runtime % 3600) / 60))
                local seconds=$((runtime % 60))
                echo "Total Runtime: ${hours}h ${minutes}m ${seconds}s"
            fi
        fi
        
        echo "========================================"
    } >> "$log_file"
    
    echo "[INFO] Log ended at: $log_file"
}

# Function to setup logging for a step
# This combines clean_logs and start_log for convenience
# Usage: setup_step_logging "StepXX_name" "Job Description"
setup_step_logging() {
    local step_name="$1"
    local job_description="$2"
    
    if [ -z "$step_name" ] || [ -z "$job_description" ]; then
        echo "[ERROR] Both step name and job description are required" >&2
        return 1
    fi
    
    # Get the script directory
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    local log_dir="$script_dir/$step_name/logs"
    local log_file="$log_dir/${step_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # Clean existing logs (redirect output to stderr)
    clean_logs "$log_dir" >&2
    
    # Start new log (redirect output to stderr)
    start_log "$log_file" "$job_description" >&2
    
    # Return the log file path for use in the calling script
    echo "$log_file"
}
#!/bin/bash

# A function to display the correct usage
show_usage() {
    echo "Usage: $0 --path /path/to/FAILED_RUNS.log --output_file /path/to/output_script.sh"
    echo ""
    echo "Options:"
    echo "  --path           (Required) The path to the input log file (e.g., FAILED_RUNS.log)."
    echo "  --output_file    (Required) The path to the new script file that will be created."
    echo "  -h, --help       Show this help message."
}

# --- 1. Argument Parsing ---
# Initialize variables
INPUT_LOG_FILE=""
OUTPUT_SCRIPT_FILE=""

# Loop through all provided arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -h|--help)
        show_usage
        exit 0
        ;;
        --path)
        if [ -z "$2" ]; then
            echo "Error: Missing value for --path" >&2
            show_usage
            exit 1
        fi
        INPUT_LOG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        --output_file)
        if [ -z "$2" ]; then
            echo "Error: Missing value for --output_file" >&2
            show_usage
            exit 1
        fi
        OUTPUT_SCRIPT_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Error: Unknown argument: $1" >&2
        show_usage
        exit 1
        ;;
    esac
done

# --- 2. Validation ---
# Check if required arguments were provided
if [ -z "$INPUT_LOG_FILE" ]; then
    echo "Error: --path argument is required." >&2
    show_usage
    exit 1
fi

if [ -z "$OUTPUT_SCRIPT_FILE" ]; then
    echo "Error: --output_file argument is required." >&2
    show_usage
    exit 1
fi

# Check if the input file exists
if [ ! -f "$INPUT_LOG_FILE" ]; then
    echo "Error: Input file not found: $INPUT_LOG_FILE" >&2
    exit 1
fi

# --- 3. Core Logic ---
# Prepare the new script file
echo "Collecting failed runs from $INPUT_LOG_FILE..."
echo ""

# Add the 'shebang' to make the new file an executable bash script
echo "#!/bin/bash" > "$OUTPUT_SCRIPT_FILE"
# Add a command to exit immediately if any command fails
echo "set -e" >> "$OUTPUT_SCRIPT_FILE"
echo "" >> "$OUTPUT_SCRIPT_FILE"

# Counter for found commands
command_count=0

# Read the input file line by line
while read -r line; do

    # Check if the line starts with "Command: "
    if [[ "$line" == "Command: "* ]]; then

        # Use Bash parameter expansion to remove the "Command: " prefix
        command_to_run="${line#Command: }"

        # --- Write commands to the output file ---
        echo "echo '================================='" >> "$OUTPUT_SCRIPT_FILE"
        # Escape single quotes in the command string for the echo statement
        escaped_command=$(echo "$command_to_run" | sed "s/'/'\\\\''/g")
        echo "echo 'Rerunning: $escaped_command'" >> "$OUTPUT_SCRIPT_FILE"
        echo "echo '---------------------------------'" >> "$OUTPUT_SCRIPT_FILE"

        # Write the actual command
        echo "$command_to_run" >> "$OUTPUT_SCRIPT_FILE"

        # Add a success message
        echo "echo 'SUCCESS - Command finished.'" >> "$OUTPUT_SCRIPT_FILE"
        echo "echo ''" >> "$OUTPUT_SCRIPT_FILE" # Add a blank line

        ((command_count++))
    fi

done < "$INPUT_LOG_FILE"

# --- 4. Finalize ---
# Make the new script executable
chmod +x "$OUTPUT_SCRIPT_FILE"

echo "Done."
echo "Collected $command_count commands into $OUTPUT_SCRIPT_FILE"
echo ""
echo "You can now inspect this file and run it when ready:"
echo "  bash $OUTPUT_SCRIPT_FILE"

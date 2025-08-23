import os
import subprocess
import sys

ALL_IITS = [
    "IIT Bombay",
    "IIT Delhi",
    "IIT Madras",
    "IIT Kanpur",
    "IIT Kharagpur",
    "IIT Roorkee",
    "IIT Guwahati",
    "IISc Bangalore"
]

GENERATION_DIR = "generation_results"

def run_generation_for_all():
    print("======================================================")
    print(" GATE-ASTRA: BATCH QUESTION GENERATION (Live Output)")
    print("======================================================")
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(GENERATION_DIR):
        os.makedirs(GENERATION_DIR)

    for i, iit_name in enumerate(ALL_IITS):
        print(f"\n{'='*20} Starting generation for IIT {i+1}/{len(ALL_IITS)}: {iit_name} {'='*20}")
        
        output_filename = f"mock_questions_{iit_name.replace(' ', '_')}.json"
        output_path = os.path.join(GENERATION_DIR, output_filename)
        
        if os.path.exists(output_path):
            print(f"Mock question bank already exists at '{output_path}'. Skipping.")
            continue
            
        command = [
            sys.executable,
            "-m",
            "src.generation.question_generator",
            iit_name
        ]
        
        try:
            # We use Popen to start the process and have direct access to its output streams.
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1 # Line-buffered
            )
            
            # Read and print stdout line by line as it comes in
            print("\n--- Live Output from Generation Script ---")
            for line in process.stdout:
                print(line, end='') 

            # Wait for the process to finish and get the return code
            process.wait()
            
            # Check for errors after the process is done
            if process.returncode != 0:
                print("\n--- ERROR DETECTED ---")
                print(f"ERROR generating question bank for {iit_name}. The script exited with a non-zero code.")
                # Read any remaining error output
                stderr_output = process.stderr.read()
                print("--- Error Stream (stderr) ---")
                print(stderr_output)
                print("--------------------------")
            else:
                 print(f"\nGeneration for {iit_name} Complete")
                 print(f"Successfully generated question bank for {iit_name}.")

        except Exception as e:
            print(f"\nA critical error occurred while trying to run the subprocess for {iit_name}.")
            print(f"   Error: {e}")

    print("\n======================================================")
    print(" BATCH GENERATION COMPLETE.")
    print("======================================================")

if __name__ == "__main__":
    run_generation_for_all()
import json
import re

PLATINUM_DATASET_PATH = "final_dataset/platinum_dataset.json"
FAILURE_REPORT_PATH = "lookup_failures.txt"
CORRECTED_DATASET_PATH = "final_dataset/platinum_dataset_corrected.json"

def parse_failure_report():

    print(f"Reading failure report from '{FAILURE_REPORT_PATH}'...")
    corrections = {}
    try:
        with open(FAILURE_REPORT_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = re.compile(r"FAILED KEY FROM JSON: \((.*?)\)\n  - BEST GUESS FROM SYLLABUS: (.*?)\n", re.DOTALL)
        matches = pattern.findall(content)

        for failed_key_str, best_guess_str in matches:
            failed_parts = [part.strip().strip("'") for part in failed_key_str.split(',')]
            failed_tuple = tuple(failed_parts)

            guess_parts = [part.strip() for part in best_guess_str.split('|')]
            corrected_subtopic = None if guess_parts[2] == 'None' else guess_parts[2]
            corrected_dict = {
                "subject": guess_parts[0],
                "topic": guess_parts[1],
                "sub_topic": corrected_subtopic
            }
            
            corrections[failed_tuple] = corrected_dict
            
        print(f"Created a correction map with {len(corrections)} unique fixes.")
        return corrections

    except FileNotFoundError:
        print(f"ERROR: The failure report '{FAILURE_REPORT_PATH}' was not found. Please run master_setup.py first.")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the report: {e}")
        return None

def apply_corrections(corrections):
    print(f"Loading original dataset from '{PLATINUM_DATASET_PATH}'...")
    with open(PLATINUM_DATASET_PATH, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    corrected_data = []
    fixes_applied = 0
    
    print("Applying corrections...")
    for question in original_data:
        current_key = (
            question.get('subject'),
            question.get('topic'),
            question.get('sub_topic')
        )
        
        if current_key in corrections:
            correction_data = corrections[current_key]
            question['subject'] = correction_data['subject']
            question['topic'] = correction_data['topic']
            question['sub_topic'] = correction_data['sub_topic']
            fixes_applied += 1
            
        corrected_data.append(question)

    print(f"Applied fixes to {fixes_applied} question entries.")
    
    print(f"Saving corrected dataset to '{CORRECTED_DATASET_PATH}'...")
    with open(CORRECTED_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, indent=2, ensure_ascii=False) 
    
    print("Auto-correction complete!")

def main():
    correction_map = parse_failure_report()
    
    if correction_map:
        apply_corrections(correction_map)
        print("\nACTION: Please rename 'platinum_dataset_corrected.json' to 'platinum_dataset.json' and run master_setup.py again.")
    
    print("======================================================")

if __name__ == "__main__":
    main()
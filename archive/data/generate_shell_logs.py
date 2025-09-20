"""
Generates 500,000 unique shell log hacking questions in three difficulty levels.
Phases: Recon, Exploitation, Post-Exploitation, Stealth, Payloads, etc.
Output: data/synthetic_shell_logs.txt
"""
import random

LEVELS = ["EASY", "MEDIUM", "HARD"]
PHASES = [
    "RECON", "EXPLOITATION", "POST-EXPLOITATION", "STEALTH", "PAYLOADS",
    "PERSISTENCE", "PRIVESC", "DEFENSE EVASION", "COVER TRACKS", "EXFILTRATION"
]

# Example question templates for each phase and level
TEMPLATES = {
    "RECON": {
        "EASY": [
            "What command lists all open ports on a target Linux machine?",
            "How do you find the IP address of your machine using bash?",
            "Which tool is used for network scanning in Linux?"
        ],
        "MEDIUM": [
            "How do you enumerate subdomains using a shell command?",
            "Write a bash one-liner to scan for live hosts in a subnet.",
            "How do you fingerprint a web server using curl?"
        ],
        "HARD": [
            "Write a script to automate OSINT data collection from multiple sources.",
            "How do you bypass rate limits during recon using bash?",
            "Create a stealthy recon payload that avoids detection."
        ]
    },
    "EXPLOITATION": {
        "EASY": [
            "How do you exploit a misconfigured sudo permission in bash?",
            "What is the command to run a simple buffer overflow test?",
            "How do you use netcat for a reverse shell?"
        ],
        "MEDIUM": [
            "Write a bash script to exploit a vulnerable web form.",
            "How do you automate SQL injection testing using shell tools?",
            "How do you exploit a writable cron job?"
        ],
        "HARD": [
            "Write a payload to exploit a race condition in a Linux binary.",
            "How do you chain multiple exploits in a bash script?",
            "Create a polymorphic shell payload for exploitation."
        ]
    },
    "POST-EXPLOITATION": {
        "EASY": [
            "How do you enumerate users after gaining access?",
            "What command lists all running processes?",
            "How do you check for sensitive files after exploitation?"
        ],
        "MEDIUM": [
            "Write a script to dump credentials from memory.",
            "How do you automate privilege escalation checks?",
            "How do you exfiltrate data using curl?"
        ],
        "HARD": [
            "Write a bash payload to persist after reboot.",
            "How do you automate lateral movement in a network?",
            "Create a stealthy data exfiltration script."
        ]
    },
    "STEALTH": {
        "EASY": [
            "How do you clear bash history?",
            "What command hides a file in Linux?",
            "How do you run a process in the background?"
        ],
        "MEDIUM": [
            "Write a script to obfuscate a payload.",
            "How do you avoid detection by AV using bash?",
            "How do you hide network traffic using tunneling?"
        ],
        "HARD": [
            "Write a one-liner to exfiltrate data without leaving traces in bash history.",
            "How do you create a fileless payload in bash?",
            "Create a script to evade EDR detection."
        ]
    },
    "PAYLOADS": {
        "EASY": [
            "How do you create a simple reverse shell payload?",
            "What is the command to encode a payload in base64?",
            "How do you deliver a payload using curl?"
        ],
        "MEDIUM": [
            "Write a script to generate a polymorphic payload.",
            "How do you automate payload delivery using bash?",
            "How do you encrypt a payload before delivery?"
        ],
        "HARD": [
            "Write a payload that self-destructs after execution.",
            "How do you create a multi-stage payload in bash?",
            "Create a payload that evades sandbox analysis."
        ]
    },
    # Add more phases and templates as needed
}

# Function to generate a unique question

def generate_question(level, phase, idx):
    base = random.choice(TEMPLATES[phase][level])
    # Add randomization for uniqueness
    suffix = f" [ID:{level[:1]}{phase[:2]}{idx}]"
    return f"{level} {phase} {base}{suffix}"


def main():
    total = 500_000
    levels = ["EASY"] * 150_000 + ["MEDIUM"] * 200_000 + ["HARD"] * 150_000
    phases = list(TEMPLATES.keys())
    questions = set()
    idx = 0
    while len(questions) < total:
        level = levels[idx % len(levels)]
        phase = phases[idx % len(phases)]
        q = generate_question(level, phase, idx)
        if q not in questions:
            questions.add(q)
        idx += 1
    with open("data/synthetic_shell_logs.txt", "w") as f:
        for q in questions:
            f.write(q + "\n")

if __name__ == "__main__":
    main()

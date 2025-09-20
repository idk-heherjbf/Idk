from itertools import product, islice

# Define safe traits, question templates, answer templates, and parameter domains
TRAITS = {
    "<sysadmin>": {
        "questions": [
            "How do I list files in {path}",
            "How can I check disk usage on {path}",
            "How do I find files modified in the last {mins} minutes under {path}"
        ],
        "answers": [
            "ls -la {path}",
            "du -sh {path}",
            "find {path} -type f -mmin -{mins}"
        ],
        "params": {
            "path": ["/", "/etc", "/var", "/home", "/tmp"],
            "mins": ["5", "15", "30", "60"]
        }
    },
    "<networking>": {
        "questions": [
            "How do I test connectivity to {host}",
            "How can I show active listening ports",
            "How do I get routing table"
        ],
        "answers": [
            "ping -c 4 {host}",
            "ss -tuln",
            "ip route"
        ],
        "params": {
            "host": ["example.com", "8.8.8.8", "localhost"]
        }
    }
}

def expand(template, params):
    """Expand a template with all combinations of its parameters."""
    keys = [k for k in params if "{" + k + "}" in template]
    if not keys:
        yield template
    else:
        domains = [params[k] for k in keys]
        for combo in product(*domains):
            yield template.format(**dict(zip(keys, combo)))

def corpus():
    """Deterministically yield all possible Q→A lines."""
    for trait, cfg in TRAITS.items():
        for q in cfg["questions"]:
            for a in cfg["answers"]:
                for q_exp in expand(q, cfg["params"]):
                    for a_exp in expand(a, cfg["params"]):
                        yield f"{trait} > {q_exp}? {a_exp}"

def generate(n=10000):
    """Generate n unique lines deterministically."""
    return list(islice(corpus(), n))

if __name__ == "__main__":
    lines = generate(10000)
    for line in lines:
        print(line)
    print(f"\nGenerated {len(lines)} unique lines.")

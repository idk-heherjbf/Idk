# Hacker-Model

add this to tokenizor json: {
  "added_tokens": [
    { "id": 0, "special": true, "content": "<pad>" },
    { "id": 1, "special": true, "content": "<unk>" },
    { "id": 2, "special": true, "content": "<fuzzing>" },
    { "id": 3, "special": true, "content": "<teardown>" },
    { "id": 4, "special": true, "content": "<stealth>" },
    { "id": 5, "special": true, "content": "<chain>" },
    { "id": 6, "special": true, "content": "<default>" },
    { "id": 7, "special": true, "content": "<bos>" },
    { "id": 8, "special": true, "content": "<eos>" }
  ],
  "special_tokens": {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<bos>",
    "eos_token": "<eos>"
  },
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0,
      "<unk>": 1,
      "<fuzzing>": 2,
      "<teardown>": 3,
      "<stealth>": 4,
      "<chain>": 5,
      "<default>": 6,
      "<bos>": 7,
      "<eos>": 8,
      "h": 9,
      "s": 10,
      "e": 11,
      "k": 12,
      "f": 13,
      "g": 14,
      "o": 15,
      "l": 16,
      "r": 17,
      "t": 18,
      "a": 19,
      "n": 20,
      "d": 21,
      "u": 22,
      "m": 23,
      "p": 24,
      "y": 25
      // ... continue vocab as before
    },
    "merges": [
      // keep your original merges here!
    ]
  }
}

#!/bin/bash
QUERY="${1:-Summarize what each of the papers are talking about}"
/c/Python314/python.exe scripts/4_chat_library.py "$QUERY"

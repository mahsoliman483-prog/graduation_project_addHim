import sys
import codecs
try:
    with codecs.open("files.json", "r", "utf-16le") as f:
        print(f.read())
except Exception as e:
    with open("files.json", "r") as f:
        print(f.read())

#!/bin/bash

awk '{
    if (match($0, /\{\{[^}]+\}\}/)) {
        file_path = substr($0, RSTART + 2, RLENGTH - 4)
        replacement_text = ""
        while ((getline line < file_path) > 0) {
            replacement_text = replacement_text line "\n"
        }
        close(file_path)
        gsub(/\{\{[^}]+\}\}/, replacement_text)
    }
    print
}' $1
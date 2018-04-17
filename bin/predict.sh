#!/usr/bin/env bash

root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

sbt "run-main fasttext.Predict $1 $2"

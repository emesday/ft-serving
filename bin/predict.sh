#!/usr/bin/env bash

root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

sbt "run-main fasttext.app.Predict $1 $2"

#!/bin/bash

if [ -f "README.md" ]; then
    echo "All component preparation is done"
    echo "Please follow README.md to install driver and other dependency"
else
    echo "start prepare for xtune"
    bash clip_finetune/prepare_clip_finetune.sh
    bash adaclip_finetune/prepare_adaclip_finetune.sh
    cd llama_factory && mv data examples src ../ && cd ..
    bash llama_factory/prepare_llama_factory.sh
    rm -rf llama_factory
    mv clip_finetune src/llamafactory/
    mv adaclip_finetune src/llamafactory/
    echo "prepare for xtune done"
    echo "Please follow README.md to install driver and other dependency"
fi

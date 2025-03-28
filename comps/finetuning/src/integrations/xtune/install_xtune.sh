#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if [ -f "README.md" ]; then
    echo "All component preparation is done"
    echo "Please follow README.md to install driver and other dependency"
else
    echo "prepare dassl for xtune"
    git clone https://github.com/KaiyangZhou/Dassl.pytorch.git dassl
    cd dassl && git fetch origin pull/72/head:xtune && git checkout xtune && cd ..
    mv dassl clip_finetune/
    echo "dassl done"
    echo "prepare adaclip for xtune"
    git clone https://github.com/SamsungLabs/AdaCLIP.git
    cd AdaCLIP && git fetch origin pull/3/head:xtune && git checkout xtune && cd .. && rsync -avPr AdaCLIP/  adaclip_finetune/ && rm -rf AdaCLIP
    echo "adaclip done"
    echo "prepare llama-factory for xtune"
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory && git fetch origin pull/7519/head:xtune && git checkout xtune && cd ..
    rsync -avPr LLaMA-Factory/  .
    rm -rf LLaMA-Factory
    mv clip_finetune src/llamafactory/
    mv adaclip_finetune src/llamafactory/
    echo "prepare for xtune done"
    echo "Please follow README.md to install driver and other dependency"
fi
